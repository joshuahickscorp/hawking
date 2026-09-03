//! Native Qwen3.8 hybrid token graph: Q4 GEMVs + Q80 f32 activation
//! kernels + Q38-forked rearrange/GQA. Dense SwiGLU suffix. Zero fallbacks.
//!
//! Mixed catalogs (`catalog.hq38m20`) bind HGRAVB01 / HGRAVR02 / HGRAVS01
//! and pack-declared HGRAVU01 (including MLP) through the existing Q80
//! occupancy tiles. Packed bytes stay packed. A missing codec fails the
//! run; there is no reconstruct-to-Q4 path.

use super::qwen38_64_layer_execution_schedule::qwen38_assert_schedule_intact;
use super::qwen38_geometry::{ARGMAX_GROUPS, 
    qwen38_layer_name,
    Qwen38DeltaNetLayout, Qwen38MixerKind, QWEN38_DELTANET_LAYERS, QWEN38_FULL_ATTENTION_INTERVAL,
    QWEN38_GQA_HEAD_DIM,
    QWEN38_GQA_HEADS, QWEN38_GQA_KV_HEADS, QWEN38_GQA_LAYERS, QWEN38_GQA_ROTARY_DIM,
    QWEN38_HIDDEN, QWEN38_INTERMEDIATE, QWEN38_LAYERS, QWEN38_RMS_EPS, QWEN38_ROPE_THETA,
    QWEN38_VOCAB,
};
use super::qwen38_pack::{
    load_qwen38_manifest, read_qwen38_f32_payload, QWEN38_EXPECTED_CATALOG_TENSORS,
};
use super::qwen_complete_binary::{
    expand_rice_indices, mixed_gpu_layout, parse_uniform_q4_header, rice_q1_row_ptr,
    uniform_factor_value, BinaryGroupPacked, MixedGpuKind, RiceQ1Packed, UniformFactorPacked,
    MAGIC_AFFINE, MAGIC_BINARY, MAGIC_HGRAVS01, MAGIC_RESIDUAL_COMPACT, MAGIC_UNIFORM,
    UNIFORM_Q4_GROUP_SIZE, UNIFORM_Q4_GROUP_SIZE_128, affine_group_size_supported,
    uniform_q4_group_size_supported,
};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const QWEN38_MIXED_CATALOG_MAGIC: [u8; 8] = *b"HQ38M20\0";
pub const QWEN38_MIXED_CATALOG_VERSION: u32 = 1;
pub const QWEN38_MIXED_RECORD_SIZE: usize = 128;
pub const QWEN38_MIXED_CATALOG_NAME: &str = "catalog.hq38m20";
pub const QWEN38_MIXED_SCHEMA: &str =
    "hawking.ascension.qwen38_mixed_representation_candidate.v1";
pub const QWEN38_MIXED_HGRAVS_RANK: usize = 160;
pub const QWEN38_MIXED_HGRAVS_BITS: u8 = 3;
pub const QWEN38_MIXED_HGRAVS_GROUP: usize = 64;

/// Default **on**. Consume packed mixed codes on the Q80 occupancy tiles.
/// `HAWKING_QWEN38_RECON_FUSE=0` selects the G023 serial family names.
pub fn qwen38_recon_fuse_enabled() -> bool {
    crate::env_opt_out("HAWKING_QWEN38_RECON_FUSE")
}

/// `true` only when `HAWKING_TRACE_DISPATCH=1`. Default off so
/// `MetalContext` is built with `new_with_trace(false)` — the same
/// constructor the decode path used before this lever existed.
pub fn qwen38_trace_dispatch_enabled() -> bool {
    crate::env_on("HAWKING_TRACE_DISPATCH")
}

/// Opt-in fastest-candidate profile for the resident Qwen38 token graph.
///
/// This is deliberately a profile switch rather than a silent production
/// default.  Every individual lever still has an environment override, so a
/// protected A/B can hold one candidate out while enabling the rest:
/// `HAWKING_QWEN38_FAST=1` supplies the measured starting point and an
/// explicit `HAWKING_*` value wins over the profile default.
pub fn qwen38_fast_profile_enabled() -> bool {
    crate::env_on("HAWKING_QWEN38_FAST")
}

fn fast_default_bool(name: &str, fast: bool) -> bool {
    match std::env::var_os(name) {
        Some(_) => crate::env_on(name),
        None => fast,
    }
}

#[inline]
fn qwen38_family_kernel(family_dispatch: bool, family: &'static str, legacy: &'static str) -> &'static str {
    if family_dispatch {
        family
    } else {
        legacy
    }
}

#[inline]
fn qwen38_binary_family_kernel(family_dispatch: bool) -> &'static str {
    qwen38_family_kernel(
        family_dispatch,
        crate::decode_family::MATVEC_BINARY,
        crate::decode_family::LEGACY_MATVEC_BINARY,
    )
}

#[inline]
fn qwen38_hgravs_family_kernel(family_dispatch: bool) -> &'static str {
    qwen38_family_kernel(
        family_dispatch,
        crate::decode_family::MATVEC_HGRAVS,
        crate::decode_family::LEGACY_MATVEC_HGRAVS,
    )
}

#[inline]
fn qwen38_swiglu_family_kernel(family_dispatch: bool) -> &'static str {
    qwen38_family_kernel(
        family_dispatch,
        crate::decode_family::SWIGLU_F32,
        crate::decode_family::LEGACY_SWIGLU_F32,
    )
}

fn validated_threadgroup_env(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value == 0 || (value.is_power_of_two() && (32..=1024).contains(value)))
        .unwrap_or(default)
}

/// Geometry is resolved once per resident session, not once per layer on
/// every token. The defaults are the already-measured retiled candidates;
/// zero keeps each historical scalar control available for A/B runs.
fn qwen38_rmsnorm_tg_from_env() -> u32 {
    validated_threadgroup_env("HAWKING_RMSNORM_TG", 1024)
}

fn qwen38_dn_rmsnorm_tg_from_env() -> u32 {
    validated_threadgroup_env("HAWKING_DN_RMSNORM_TG", 256)
}

fn qwen38_rope_tg_from_env() -> u32 {
    validated_threadgroup_env("HAWKING_ROPE_TG", 256)
}

/// MLP suffix fusion. Default Off keeps the 964-dispatch production graph.
///
/// `HAWKING_QWEN38_FUSE_MLP=pair`          — gate+up in one geo_tpr64 dispatch (still SwiGLU)
/// `HAWKING_QWEN38_FUSE_MLP=swiglu` or `=1` — gate+up+SwiGLU in one dispatch
/// anything else                            — PANICS, see `from_env`
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Qwen38MlpFusion {
    #[default]
    Off,
    GateUpPair,
    GateUpSwiglu,
}

impl Qwen38MlpFusion {
    /// Unset means Off. An UNRECOGNISED value PANICS rather than silently
    /// meaning Off.
    ///
    /// Why this is not merely defensive: the three sibling levers
    /// (`FUSE_GQA_QKV`, `FUSE_DN_INPROJ`, `FUSE_ADD_RMSNORM`) are all
    /// `=1` flags, so `=1` is the natural thing to write here too -- and it
    /// used to parse to `Off`. A measurement run that way records the
    /// UNFUSED graph while its operator believes the lever is on, and the
    /// dispatch count comes back unchanged, which reads as "this lever is
    /// inert" rather than "this lever never ran". That misreading was
    /// published once (receipts/headless/TOKEN_EXECUTION_ATLAS_COUNTS.json,
    /// corrected by ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json). `1`/`true`/
    /// `on`/`yes` therefore mean the STRONGEST fusion, matching what `=1`
    /// means for every sibling lever.
    pub fn from_env() -> Self {
        Self::from_env_with_fast(qwen38_fast_profile_enabled())
    }

    fn from_env_with_fast(fast: bool) -> Self {
        match std::env::var("HAWKING_QWEN38_FUSE_MLP") {
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "" | "0" | "off" | "false" | "no" => Self::Off,
                "pair" | "gate_up" => Self::GateUpPair,
                "swiglu" | "gate_up_swiglu" | "1" | "true" | "on" | "yes" => {
                    Self::GateUpSwiglu
                }
                other => panic!(
                    "HAWKING_QWEN38_FUSE_MLP={other:?} is not a recognised value; \
                     use pair | swiglu | 1 | 0. Falling back to Off here would \
                     silently measure the unfused graph."
                ),
            },
            Err(_) => {
                if fast {
                    Self::GateUpSwiglu
                } else {
                    Self::Off
                }
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::GateUpPair => "pair",
            Self::GateUpSwiglu => "swiglu",
        }
    }

    /// Dispatches removed from the 6-kernel MLP suffix, per token (64 layers).
    pub fn saved_dispatches_per_token(self) -> u64 {
        match self {
            Self::Off => 0,
            Self::GateUpPair => QWEN38_LAYERS as u64,
            Self::GateUpSwiglu => 2 * QWEN38_LAYERS as u64,
        }
    }
}

pub fn qwen38_concurrent_independent_enabled() -> bool {
    crate::env_on("HAWKING_QWEN38_CONCURRENT")
}

pub fn qwen38_fuse_gqa_qkv_enabled() -> bool {
    fast_default_bool("HAWKING_QWEN38_FUSE_GQA_QKV", qwen38_fast_profile_enabled())
}

pub fn qwen38_fuse_dn_inproj_enabled() -> bool {
    fast_default_bool(
        "HAWKING_QWEN38_FUSE_DN_INPROJ",
        qwen38_fast_profile_enabled(),
    )
}

pub fn qwen38_serial_token_encoder_enabled() -> bool {
    fast_default_bool(
        "HAWKING_QWEN38_SERIAL_TOKEN_ENCODER",
        qwen38_fast_profile_enabled(),
    )
}

/// Fuse the Qwen3.8 per-head attention sigmoid gate into MHA's final write.
/// The fast profile enables this candidate; `=0` restores the historical
/// MHA-then-gate graph for direct A/B comparison.
pub fn qwen38_fuse_attention_gate_enabled() -> bool {
    fast_default_bool(
        "HAWKING_QWEN38_FUSE_ATTENTION_GATE",
        qwen38_fast_profile_enabled(),
    )
}

/// Residual add + the following RMSNorm. Default Off.
///
/// `HAWKING_QWEN38_FUSE_ADD_RMSNORM=1`   — (1+w) production math
/// `HAWKING_QWEN38_FUSE_ADD_RMSNORM=bad` — plain `weight[i]` BAD control
pub fn qwen38_fuse_add_rmsnorm_from_env() -> (bool, bool) {
    qwen38_fuse_add_rmsnorm_from_env_with_fast(qwen38_fast_profile_enabled())
}

fn qwen38_fuse_add_rmsnorm_from_env_with_fast(fast: bool) -> (bool, bool) {
    match std::env::var("HAWKING_QWEN38_FUSE_ADD_RMSNORM") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => (true, false),
            "bad" | "plainweight" => (true, true),
            _ => (false, false),
        },
        Err(_) => (fast, false),
    }
}

pub fn qwen38_fuse_add_rmsnorm_enabled() -> bool {
    qwen38_fuse_add_rmsnorm_from_env().0
}

/// Mixer residual + MLP RMSNorm, and MLP residual + next mixer RMSNorm
/// (last layer: MLP residual + final norm). 2 launches saved per layer.
pub const QWEN38_ADD_RMSNORM_SAVED_PER_TOKEN: u64 = 2 * QWEN38_LAYERS as u64;
/// One standalone attention-gate launch is removed for each Qwen3.8 GQA layer.
pub const QWEN38_ATTENTION_GATE_SAVED_PER_TOKEN: u64 = QWEN38_GQA_LAYERS as u64;

/// Production 964 minus the fusions named. Counted the same way as
/// `production_dispatches_per_token` (one kernel launch = one dispatch).
pub fn qwen38_fused_dispatches_per_token(
    mlp: Qwen38MlpFusion,
    fuse_gqa_qkv: bool,
    fuse_dn_inproj: bool,
) -> u64 {
    qwen38_fused_dispatches_per_token_ex(mlp, fuse_gqa_qkv, fuse_dn_inproj, false)
}

pub fn qwen38_fused_dispatches_per_token_ex(
    mlp: Qwen38MlpFusion,
    fuse_gqa_qkv: bool,
    fuse_dn_inproj: bool,
    fuse_add_rmsnorm: bool,
) -> u64 {
    qwen38_fused_dispatches_per_token_full(
        mlp,
        fuse_gqa_qkv,
        fuse_dn_inproj,
        fuse_add_rmsnorm,
        false,
    )
}

/// ba_to_decay folded into gated-delta. 48 launches on the 628 graph.
/// Default Off — production stays 756/628 until a child enables it
/// (`HAWKING_QWEN38_FUSE_BA_DELTA=1`) or selects a fused-ba sibling
/// kernel (`HAWKING_QWEN38_DN_STATE=widen_f4`, which also folds
/// ba_to_decay because that kernel consumes projected_ba in-register).
pub const QWEN38_BA_DELTA_SAVED_PER_TOKEN: u64 = QWEN38_DELTANET_LAYERS as u64;
pub const QWEN38_BA_DELTA_KERNEL: &str = "qwen38_gated_delta_decode_vi_simd_ba";
pub const QWEN38_BA_DELTA_BAD_KERNEL: &str = "qwen38_gated_delta_decode_vi_simd_ba_plain";
pub const QWEN38_DN_STATE_F4_KERNEL: &str = "qwen38_gated_delta_decode_vi_simd_ba_f4";
pub const QWEN38_DN_STATE_TG32_KERNEL: &str = "qwen38_gated_delta_decode_vi_simd_ba_tg32";
/// 128 ki × 32 vi tile + 16-float simd partials (change 2).
pub const QWEN38_DN_STATE_TG32_BYTES: u64 = (128 * 32 + 16) * 4;

/// Gated-delta state-update kernel on the real `encode_deltanet` path.
///
/// Default `Baseline` is the unfused vi-SIMD launch on the 628-dispatch
/// sealed graph. `WidenF4` packs 4 vi as float4 (fused-ba sibling).
/// `CoalesceTg32` stages a 128×32 state tile in threadgroup memory.
/// Production stays Baseline unless a child sets
/// `HAWKING_QWEN38_DN_STATE`. WidenF4 / CoalesceTg32 consume
/// projected_ba in-register, so they fold ba_to_decay even when
/// `FUSE_BA_DELTA` is unset — otherwise the flag was a no-op and
/// production kept launching unfused vi-SIMD.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38DeltaNetStateKernel {
    Baseline,
    WidenF4,
    CoalesceTg32,
}

impl Qwen38DeltaNetStateKernel {
    pub fn fused_ba_name(self, bad: bool) -> &'static str {
        if bad {
            return QWEN38_BA_DELTA_BAD_KERNEL;
        }
        match self {
            Self::Baseline => QWEN38_BA_DELTA_KERNEL,
            Self::WidenF4 => QWEN38_DN_STATE_F4_KERNEL,
            Self::CoalesceTg32 => QWEN38_DN_STATE_TG32_KERNEL,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::WidenF4 => "widen_f4",
            Self::CoalesceTg32 => "coalesce_tg32",
        }
    }

    /// These kernels take projected_ba / A_log / dt_bias, not precomputed
    /// decay/beta. Launching them without folding ba_to_decay would both
    /// compute decay twice and keep a launch that the layout already ate.
    pub fn folds_ba_to_decay(self) -> bool {
        !matches!(self, Self::Baseline)
    }

    pub fn from_env() -> Self {
        Self::from_env_with_fast(qwen38_fast_profile_enabled())
    }

    fn from_env_with_fast(fast: bool) -> Self {
        match std::env::var("HAWKING_QWEN38_DN_STATE") {
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "f4" | "widen" | "widen_f4" => Self::WidenF4,
                "tg32" | "coalesce" | "coalesce_tg32" => Self::CoalesceTg32,
                _ => Self::Baseline,
            },
            // G126: PROMOTED. widen_f4 is the sealed default, not a fast-profile
            // opt-in. It is the CONTROL arm of the protected bitcast lease
            // (580 dispatches, token-identical), so the sealed graph and the
            // measured graph are the same graph. `fast` no longer selects it
            // because it is already on.
            Err(_) => {
                let _ = fast;
                Self::WidenF4
            }
        }
    }
}

/// `encode_dn_ba_and_delta` launches a fused-ba sibling when either the
/// explicit FUSE_BA_DELTA flag is on or the selected state kernel cannot
/// consume precomputed decay/beta.
pub fn qwen38_dn_state_uses_fused_ba(
    fuse_ba_delta: bool,
    kernel: Qwen38DeltaNetStateKernel,
) -> bool {
    fuse_ba_delta || kernel.folds_ba_to_decay()
}

/// Recurrent-state + rec_out parity of a candidate gated-delta kernel
/// against `qwen38_gated_delta_decode_vi_simd_ba` on one layer.
#[derive(Clone, Debug)]
pub struct Qwen38DeltaNetStateParity {
    pub kernel: &'static str,
    pub layer: usize,
    pub max_abs_diff_rec_out: f32,
    pub max_abs_diff_rec_state: f32,
    pub baseline_gpu_ns: Option<u64>,
    pub candidate_gpu_ns: Option<u64>,
    pub baseline_dispatches: u64,
    pub candidate_dispatches: u64,
    pub dense_w_materialized: u64,
}

/// `HAWKING_QWEN38_FUSE_BA_DELTA=1` honest formula; `=bad` identity decay/beta.
pub fn qwen38_fuse_ba_delta_from_env() -> (bool, bool) {
    qwen38_fuse_ba_delta_from_env_with_fast(qwen38_fast_profile_enabled())
}

fn qwen38_fuse_ba_delta_from_env_with_fast(fast: bool) -> (bool, bool) {
    match std::env::var("HAWKING_QWEN38_FUSE_BA_DELTA") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => (true, false),
            "bad" | "plain" | "identity" => (true, true),
            _ => (false, false),
        },
        Err(_) => (fast, false),
    }
}

pub fn qwen38_fuse_ba_delta_enabled() -> bool {
    qwen38_fuse_ba_delta_from_env().0
}

pub fn qwen38_fused_dispatches_per_token_full(
    mlp: Qwen38MlpFusion,
    fuse_gqa_qkv: bool,
    fuse_dn_inproj: bool,
    fuse_add_rmsnorm: bool,
    fuse_ba_delta: bool,
) -> u64 {
    let mut n = super::qwen38_token_ns_ledger::production_dispatches_per_token();
    n = n.saturating_sub(mlp.saved_dispatches_per_token());
    if fuse_gqa_qkv {
        n = n.saturating_sub(2 * QWEN38_GQA_LAYERS as u64);
    }
    if fuse_dn_inproj {
        n = n.saturating_sub(QWEN38_DELTANET_LAYERS as u64);
    }
    if fuse_add_rmsnorm {
        n = n.saturating_sub(QWEN38_ADD_RMSNORM_SAVED_PER_TOKEN);
    }
    if fuse_ba_delta {
        n = n.saturating_sub(QWEN38_BA_DELTA_SAVED_PER_TOKEN);
    }
    n
}

pub const QWEN38_Q4_GATE_UP_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_gate_up_geo_tpr64_tg128";
pub const QWEN38_Q4_GATE_UP_SWIGLU_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_gate_up_swiglu_geo_tpr64_tg128";
pub const QWEN38_Q4_PAIR_CONCAT_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128";
pub const QWEN38_Q4_QKV_GEO_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128";

/// bitcast siblings of the three uniform-q4 matvecs the resident dispatches.
/// Same binds, same geometry; the nibble is unpacked straight into an f32
/// mantissa so neither the int-to-float convert nor the -8 zero point runs.
/// MEASURED BIT-IDENTICAL on a real qkvz projection at 1.1444x
/// (receipts/future/Q4_BITCAST_AB.json). Default is OFF; opt in with
/// HAWKING_Q4_UNPACK=bitcast.
pub const QWEN38_Q4_MATVEC_BITCAST: &str =
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_bitcast";
pub const QWEN38_Q4_QKV_GEO_BITCAST: &str =
    "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128_bitcast";
pub const QWEN38_Q4_PAIR_CONCAT_BITCAST: &str =
    "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128_bitcast";

/// Whether the q4 bitcast unpack is selected. Read once per call rather than
/// cached, matching how HAWKING_AFFINE2_GEO is read: a lever that cannot be
/// turned off inside one process is a lever that cannot be A/B'd.
/// G126: PROMOTED. Bitcast is the sealed default; the env var now turns it OFF
/// rather than on. An unset var must return the MEASURED arm, or the sealed
/// default reports the old number under a new label.
pub fn qwen38_q4_bitcast_on() -> bool {
    match std::env::var("HAWKING_Q4_UNPACK").as_deref() {
        Ok("bitcast") | Ok("mantissa") => true,
        Ok(_) => false,
        Err(_) => true,
    }
}

/// Production name, or its bitcast sibling when the lever is on. Any q4 matvec
/// name that has no bitcast sibling is returned unchanged rather than having
/// "_bitcast" appended, because naming a kernel that does not exist binds a
/// pipeline that fails at launch - the defect SplitK4Vec still carries.
pub fn qwen38_q4_kernel(name: &'static str) -> &'static str {
    if !qwen38_q4_bitcast_on() {
        return name;
    }
    match name {
        QWEN38_Q4_MATVEC_KERNEL => QWEN38_Q4_MATVEC_BITCAST,
        QWEN38_Q4_QKV_GEO_KERNEL => QWEN38_Q4_QKV_GEO_BITCAST,
        QWEN38_Q4_PAIR_CONCAT_KERNEL => QWEN38_Q4_PAIR_CONCAT_BITCAST,
        other => other,
    }
}
pub const QWEN38_AFFINE_GATE_UP_KERNEL: &str =
    "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128";
pub const QWEN38_AFFINE_Q2_GEO_TPR64_RUNTIME_DIV: &str =
    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_runtime_div";
pub const QWEN38_AFFINE_Q2_QMVFAST: &str = "qwen_affine_q2_group64_matvec_qmvfast_r8tg64";
pub const QWEN38_AFFINE_Q2_WIDE64: &str = "qwen_affine_q2_group64_matvec_wide64_r4tg128";
pub const QWEN38_AFFINE_Q2_TGX: &str = "qwen_affine_q2_group64_matvec_tgx_r8tg256";
pub const QWEN38_AFFINE_Q2_QMVFAST_ADDR: &str =
    "qwen_affine_q2_group64_matvec_qmvfast_r8tg64_addr_probe";
pub const QWEN38_AFFINE_GATE_UP_QMVFAST: &str =
    "qwen_affine_q2_group64_matvec_gate_up_qmvfast_r8tg64";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_QMVFAST: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_qmvfast_r8tg64";
pub const QWEN38_AFFINE_GATE_UP_WIDE64: &str =
    "qwen_affine_q2_group64_matvec_gate_up_wide64_r4tg128";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_WIDE64: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_wide64_r4tg128";
pub const QWEN38_AFFINE_GATE_UP_TGX: &str = "qwen_affine_q2_group64_matvec_gate_up_tgx_r8tg256";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_TGX: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_tgx_r8tg256";
pub const QWEN38_AFFINE_Q2_TGSB: &str = "qwen_affine_q2_group64_matvec_tgsb_tpr64_tg128";
pub const QWEN38_AFFINE_Q2_PIPE: &str = "qwen_affine_q2_group64_matvec_pipe_tpr64_tg128";
pub const QWEN38_AFFINE_Q2_SPLITK4: &str = "qwen_affine_q2_group64_matvec_splitk4_tg256";
pub const QWEN38_AFFINE_Q2_SPLITK4_VEC: &str =
    "qwen_affine_q2_group64_matvec_splitk4_vec_tg256";
pub const QWEN38_AFFINE_Q2_ACCFUSE: &str = "qwen_affine_q2_group64_matvec_accfuse_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_TGSB: &str =
    "qwen_affine_q2_group64_matvec_gate_up_tgsb_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_TGSB: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_tgsb_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_PIPE: &str =
    "qwen_affine_q2_group64_matvec_gate_up_pipe_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_PIPE: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_pipe_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_SPLITK4: &str =
    "qwen_affine_q2_group64_matvec_gate_up_splitk4_tg256";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_SPLITK4: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_splitk4_tg256";
pub const QWEN38_AFFINE_GATE_UP_SPLITK4_VEC: &str =
    "qwen_affine_q2_group64_matvec_gate_up_splitk4_vec_tg256";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_SPLITK4_VEC: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_splitk4_vec_tg256";
pub const QWEN38_AFFINE_GATE_UP_ACCFUSE: &str =
    "qwen_affine_q2_group64_matvec_gate_up_accfuse_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_ACCFUSE: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_accfuse_tpr64_tg128";
/// fold_addqx sibling of production geo_tpr64. Same occupancy and binds.
/// Default production stays Tpr64; opt-in via HAWKING_AFFINE2_GEO=fold_addqx
/// or `apply_affine2_geo(Affine2Geo::FoldAddqx)`. Reversible: unset the
/// lever and the 580-graph launches production unpack8 again.
pub const QWEN38_AFFINE_Q2_FOLD_ADDQX: &str =
    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_fold_addqx";
pub const QWEN38_AFFINE_GATE_UP_FOLD_ADDQX: &str =
    "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128_fold_addqx";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_fold_addqx";
/// bitcast sibling of production geo_tpr64. Same occupancy and binds; the
/// 2-bit code is unpacked straight into an f32 mantissa so no int-to-float
/// convert runs, and the affine is refolded per group. The op-class ablation
/// put that convert at 44% of this kernel's arithmetic
/// (receipts/future/OP_CLASS_ABLATION.json). Default production stays Tpr64;
/// opt-in via HAWKING_AFFINE2_GEO=bitcast. Reversible: unset the lever.
pub const QWEN38_AFFINE_Q2_BITCAST: &str =
    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_bitcast";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_BITCAST: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_bitcast";

pub const QWEN38_AFFINE_GATE_UP_BIASPREP: &str =
    "qwen_affine_q2_group64_matvec_gate_up_biasprep_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_BIASPREP: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128";
pub const QWEN38_AFFINE_GATE_UP_SWIGLU_BIASPREP_DROP: &str =
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128";
pub const QWEN38_RMSNORM_XSUM64_KERNEL: &str = "qwen80_residual_rmsnorm_tg_xsum64";
pub const QWEN38_ADD_RMSNORM_XSUM64_KERNEL: &str = "qwen80_add_residual_rmsnorm_tg_xsum64";
/// Group-64 x-sums of the MLP input (hidden) plus headroom for intermediate.
pub const QWEN38_XSUM64_CAP: usize = QWEN38_INTERMEDIATE / 64;

/// Affine2 GEMV launch geometry. Default is the incumbent tpr64 tile
/// (no-op control). `HAWKING_AFFINE2_GEO` selects a lever.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Affine2Geo {
    #[default]
    Tpr64,
    RuntimeDiv,
    QmvFast,
    Wide64,
    Tgx,
    /// Threadgroup-staged scale/bias (N024). Same tpr64 occupancy.
    Tgsb,
    /// Software-pipelined unpack + vectorized x (N024). Same tpr64 occupancy.
    Pipe,
    /// 4-way split-K, TG 256, 2 rows (N024). Not the N018 tgx tile.
    SplitK4,
    /// N035: split-K4 with float4 input loads and one-tile lookahead. Opt-in
    /// sibling of SplitK4; it is not selected by the fast profile yet.
    SplitK4Vec,
    /// Fuse scale/bias into the accumulate via algebraic rewrite (N024).
    AccFuse,
    /// fold_addqx unpack on the production tpr64 map. Same occupancy.
    /// Default stays Tpr64. Empirically bit-identical on sealed-3.14 MLP.
    FoldAddqx,
    /// bitcast unpack on the production tpr64 map. Same occupancy.
    Bitcast,
    /// N030: deferred group-64 bias via RMSNorm-produced x-sums. Same tpr64
    /// occupancy. Gate_up_swiglu only; single GEMVs stay tpr64.
    BiasPrep,
    /// N030 deliberately-bad control: biasprep inner loop, bias term dropped.
    BiasPrepDrop,
}

impl Affine2Geo {
    pub fn from_env() -> Self {
        Self::from_env_with_fast(qwen38_fast_profile_enabled())
    }

    fn from_value(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "runtime_div" | "bad" => Self::RuntimeDiv,
            "qmvfast" | "qmv_fast" => Self::QmvFast,
            "wide64" | "wide" => Self::Wide64,
            "tgx" | "tgx_splitk" => Self::Tgx,
            "tgsb" | "tg_scale_bias" => Self::Tgsb,
            "pipe" | "pipeline" => Self::Pipe,
            "splitk4" | "splitk" => Self::SplitK4,
            "splitk4_vec" | "splitk_vec" | "splitk4_vector" => Self::SplitK4Vec,
            "accfuse" | "acc_fuse" => Self::AccFuse,
            "fold_addqx" | "addqx" => Self::FoldAddqx,
            "bitcast" | "mantissa" => Self::Bitcast,
            "biasprep" | "xsum" | "bias_prep" => Self::BiasPrep,
            "biasprep_drop" | "dropbias" | "drop_bias" => Self::BiasPrepDrop,
            _ => Self::Tpr64,
        }
    }

    fn from_env_with_fast(fast: bool) -> Self {
        match std::env::var("HAWKING_AFFINE2_GEO") {
            Ok(v) => Self::from_value(&v),
            // G126: PROMOTED. The fast profile used to select SplitK4, which was
            // never the measured arm. Bitcast is what the protected lease timed
            // at 22.0100 ms GPU, token-identical, 0 fallbacks.
            Err(_) => {
                let _ = fast;
                Self::Bitcast
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tpr64 => "tpr64",
            Self::RuntimeDiv => "runtime_div",
            Self::QmvFast => "qmvfast",
            Self::Wide64 => "wide64",
            Self::Tgx => "tgx",
            Self::Tgsb => "tgsb",
            Self::Pipe => "pipe",
            Self::SplitK4 => "splitk4",
            Self::SplitK4Vec => "splitk4_vec",
            Self::AccFuse => "accfuse",
            Self::FoldAddqx => "fold_addqx",
            Self::Bitcast => "bitcast",
            Self::BiasPrep => "biasprep",
            Self::BiasPrepDrop => "biasprep_drop",
        }
    }

    pub fn is_g64_specialized(self) -> bool {
        matches!(
            self,
            Self::QmvFast
                | Self::Wide64
                | Self::Tgx
                | Self::Tgsb
                | Self::Pipe
                | Self::SplitK4
                | Self::SplitK4Vec
                | Self::AccFuse
        )
    }

    /// Fused gate_up_swiglu consumes group-64 x-sums written by RMSNorm.
    pub fn uses_xsum(self) -> bool {
        matches!(self, Self::BiasPrep | Self::BiasPrepDrop)
    }
}

/// Q2F geometry is independently overridable from the affine2 geometry.
/// When unset it intentionally inherits `HAWKING_AFFINE2_GEO` (and therefore
/// the historical fast-profile behavior), while an explicit value lets a
/// protected A/B isolate Q2F split-K from HGRAVF affine2 work.
pub fn qwen38_q2f_geo_from_env() -> Affine2Geo {
    match std::env::var("HAWKING_Q2F_GEO") {
        Ok(value) => Affine2Geo::from_value(&value),
        Err(_) => Affine2Geo::from_env(),
    }
}
pub const QWEN38_ADD_RMSNORM_KERNEL: &str = "qwen80_add_residual_rmsnorm_tg";
pub const QWEN38_ADD_RMSNORM_BAD_KERNEL: &str = "qwen80_add_residual_rmsnorm_tg_plainweight";

/// Component parity of a fused kernel against the unfused path.
#[derive(Clone, Debug)]
pub struct Qwen38FusionParity {
    pub fusion: &'static str,
    pub layer: usize,
    pub unfused_dispatches: u64,
    pub fused_pair_dispatches: u64,
    pub fused_swiglu_dispatches: u64,
    pub unfused_gpu_ns: Option<u64>,
    pub fused_pair_gpu_ns: Option<u64>,
    pub fused_swiglu_gpu_ns: Option<u64>,
    pub max_abs_diff_gate: f32,
    pub max_abs_diff_up: f32,
    pub max_abs_diff_act: f32,
    pub dense_w_materialized: u64,
}

fn mixed_error(message: impl Into<String>) -> Error {
    Error::Model(format!("qwen38 mixed hybrid decode: {}", message.into()))
}

fn mlx_residual_norm_to_delta_named(name: &str, values: &mut [f32]) {
    let convert = name.ends_with("input_layernorm.weight")
        || name.ends_with("post_attention_layernorm.weight")
        || name.ends_with("model.norm.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight");
    if !convert {
        return;
    }
    for value in values {
        *value -= 1.0;
    }
}

/// Destination lane for one HQ38M20 row. Packed GEMVs stay packed.
/// Codec 4 is already HF-δ f32v2 and must not be mlx-delta'd.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedCatalogLane {
    Packed(u8),
    Hq30Uq4,
    F32v2,
    HgravuVector,
    Affine,
}

/// CPU census of a mixed catalog. Does not open Metal and does not expand
/// rice indices. `expanded_to_q4` / `expanded_to_float_gemv` stay zero on
/// this path; they exist so a later reader cannot miss a forbidden fallback.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MixedCatalogCensus {
    pub tensors: usize,
    pub binary: usize,
    pub residual: usize,
    pub hgravs: usize,
    pub uniform: usize,
    pub affine: usize,
    pub q4: usize,
    pub f32: usize,
    pub refused: usize,
    pub expanded_to_q4: usize,
    pub expanded_to_float_gemv: usize,
    /// GEMV weight tensors reconstructed to dense float. Production affine/q2f
    /// upload never increments this; it exists so a reconstruct cannot hide.
    pub dense_w_materialized: usize,
    pub refusals: Vec<String>,
}

const HQ30UQ4_MAGIC: [u8; 8] = *b"HQ30UQ4\0";
const K_COMPLETE_TILE_COLS: u32 = 256;

pub fn qwen38_binary_matvec_kernel(cols: u32) -> &'static str {
    if cols > 2048 {
        "q80_binary_group_matvec_simd_bytes"
    } else {
        "q80_binary_group_matvec_tg256"
    }
}

pub fn qwen38_residual_matvec_kernel(cols: u32) -> &'static str {
    if cols > 2048 {
        "q80_binary_group_csr_matvec_bytes"
    } else {
        "q80_binary_group_csr_matvec_tg256"
    }
}

/// simd_bytes tiles 256 columns. A non-multiple would drop a remainder —
/// refuse rather than silently emit a partial-K GEMV.
pub fn qwen38_assert_k_complete_cols(cols: u32) -> Result<()> {
    if cols > 2048 && cols % K_COMPLETE_TILE_COLS != 0 {
        return Err(mixed_error(format!(
            "cols={cols} is not a {K_COMPLETE_TILE_COLS}-col tile multiple; \
             simd_bytes would drop a remainder. Refusing partial-K bind."
        )));
    }
    Ok(())
}

pub fn qwen38_mixed_k_complete_bind_message() -> String {
    let fuse = if qwen38_recon_fuse_enabled() {
        "ON"
    } else {
        "OFF"
    };
    format!(
        "qwen38-decode mixed bind: K-complete; recon_fuse={fuse} \
         uses q80_binary_group_matvec_simd_bytes / q80_binary_group_csr_matvec_bytes \
         when cols>2048 (256-col tiles; this model K in {{5120,6144}}); \
         cols<=2048 stay on tg256; recon_fuse=0 walks every column via {}",
        crate::decode_family::matvec_binary()
    )
}

fn hgravu_is_vector(name: &str, shape: &[usize]) -> bool {
    if name.ends_with("embed_tokens.weight") || name.ends_with("lm_head.weight") {
        return false;
    }
    let gemv = name.ends_with("mlp.gate_proj.weight")
        || name.ends_with("mlp.up_proj.weight")
        || name.ends_with("mlp.down_proj.weight")
        || name.ends_with("self_attn.q_proj.weight")
        || name.ends_with("self_attn.k_proj.weight")
        || name.ends_with("self_attn.v_proj.weight")
        || name.ends_with("self_attn.o_proj.weight")
        || name.contains("linear_attn.in_proj")
        || name.ends_with("linear_attn.out_proj.weight");
    if gemv {
        return false;
    }
    let elements = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b));
    matches!(elements, Some(n) if n <= 65_536)
}

pub fn classify_qwen38_mixed_payload(
    codec: u8,
    payload: &[u8],
    name: &str,
    shape: &[usize],
) -> Result<MixedCatalogLane> {
    match codec {
        0 | 1 | 2 => Ok(MixedCatalogLane::Packed(codec)),
        3 => {
            if payload.len() >= 8 && payload[..8] == MAGIC_UNIFORM {
                if hgravu_is_vector(name, shape) {
                    Ok(MixedCatalogLane::HgravuVector)
                } else {
                    Ok(MixedCatalogLane::Packed(3))
                }
            } else if payload.len() >= 8 && payload[..8] == HQ30UQ4_MAGIC {
                Ok(MixedCatalogLane::Hq30Uq4)
            } else {
                Err(mixed_error(format!(
                    "{name} codec 3 magic {:?} is not HGRAVU01/HQ30UQ4; refusing silent fallback",
                    payload.get(..8)
                )))
            }
        }
        4 => {
            // Validate the f32v2 envelope now so a short payload refuses at
            // classify time, not after a later rmsnorm miss.
            let _ = read_qwen38_f32_payload(payload)?;
            if let Some(n) = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b)) {
                let got = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                if got != n {
                    return Err(mixed_error(format!(
                        "{name} f32v2 numel {got} != shape product {n}"
                    )));
                }
            }
            Ok(MixedCatalogLane::F32v2)
        }
        5 => {
            if payload.len() >= 8 && payload[..8] == MAGIC_AFFINE {
                Ok(MixedCatalogLane::Affine)
            } else {
                Err(mixed_error(format!(
                    "{name} codec 5 magic {:?} is not HGRAVF01; refusing silent fallback",
                    payload.get(..8)
                )))
            }
        }
        other => Err(mixed_error(format!(
            "{name} unknown mixed codec {other}; refusing silent fallback"
        ))),
    }
}

/// Packed MLP kinds `load_mixed` may bind. Uniform is HGRAVU01 when the
/// pack declares it. HQ30UQ4 / f32 / missing stay outside this set so the
/// lock can still refuse a reconstruct or an unsupported role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedMlpNativeKind {
    Binary,
    Residual,
    Hgravs,
    Uniform,
    AffineScaleBias,
}

pub fn mixed_mlp_native_kind_from_lane(lane: MixedCatalogLane) -> Option<MixedMlpNativeKind> {
    match lane {
        MixedCatalogLane::Packed(0) => Some(MixedMlpNativeKind::Binary),
        MixedCatalogLane::Packed(1) => Some(MixedMlpNativeKind::Residual),
        MixedCatalogLane::Packed(2) => Some(MixedMlpNativeKind::Hgravs),
        MixedCatalogLane::Packed(3) => Some(MixedMlpNativeKind::Uniform),
        MixedCatalogLane::Affine => Some(MixedMlpNativeKind::AffineScaleBias),
        MixedCatalogLane::Packed(_)
        | MixedCatalogLane::Hq30Uq4
        | MixedCatalogLane::F32v2
        | MixedCatalogLane::HgravuVector => None,
    }
}

fn mixed_mlp_role_allowed(suffix: &str, kind: MixedMlpNativeKind) -> bool {
    // Native packed codes on any MLP GEMV are legal. mix_c (all-binary g64)
    // and N036 organ islands (binary body + HGRAVF01 q2f on a subset) both
    // execute without reconstructing dense W. The lock still refuses
    // Residual-on-gate (wrong mixed-2p0 role) and missing/Q4/f32 fallback.
    match suffix {
        "mlp.gate_proj.weight" => matches!(
            kind,
            MixedMlpNativeKind::Binary
                | MixedMlpNativeKind::Uniform
                | MixedMlpNativeKind::AffineScaleBias
        ),
        "mlp.up_proj.weight" => matches!(
            kind,
            MixedMlpNativeKind::Binary
                | MixedMlpNativeKind::Residual
                | MixedMlpNativeKind::Uniform
                | MixedMlpNativeKind::AffineScaleBias
        ),
        "mlp.down_proj.weight" => matches!(
            kind,
            MixedMlpNativeKind::Binary
                | MixedMlpNativeKind::Hgravs
                | MixedMlpNativeKind::Uniform
                | MixedMlpNativeKind::AffineScaleBias
        ),
        _ => false,
    }
}

/// CPU-side MLP role lock. Mixed-2p0 assignment (gate Binary / up Residual /
/// down Hgravs) still passes. All-binary (mix_c) and binary+affine islands
/// (N036) pass. Pack-declared Uniform or Affine passes on any of those three
/// roles. Anything else — missing, HQ30UQ4, f32, Residual-on-gate — refuses.
pub fn assert_mixed_mlp_native_kinds(
    lookup: impl Fn(&str) -> Option<MixedMlpNativeKind>,
) -> Result<()> {
    const ROLES: [(&str, &str); 3] = [
        ("mlp.gate_proj.weight", "HGRAVB01"),
        ("mlp.up_proj.weight", "HGRAVR02"),
        ("mlp.down_proj.weight", "HGRAVS01"),
    ];
    for layer in 0..QWEN38_LAYERS {
        for (suffix, label) in ROLES {
            let name = qwen38_layer_name(layer, suffix);
            match lookup(&name) {
                Some(kind) if mixed_mlp_role_allowed(suffix, kind) => {}
                Some(_) => {
                    return Err(mixed_error(format!(
                        "{name} is not {label} or HGRAVU01 or HGRAVF01; refusing reconstructed MLP"
                    )))
                }
                None => {
                    return Err(mixed_error(format!(
                        "missing {name}; refusing silent dense/Q4 fallback"
                    )))
                }
            }
        }
    }
    Ok(())
}

fn read_catalog_prefix(row: &Qwen38MixedCatalogRow, n: usize) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(&row.segment_path).map_err(|error| {
        mixed_error(format!(
            "cannot open {}: {error}",
            row.segment_path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(row.offset))
        .map_err(|error| mixed_error(format!("seek {}: {error}", row.name)))?;
    let take = n.min(usize::try_from(row.nbytes).unwrap_or(0));
    let mut prefix = vec![0u8; take];
    file.read_exact(&mut prefix)
        .map_err(|error| mixed_error(format!("read {}: {error}", row.name)))?;
    Ok(prefix)
}

fn is_mixed_mlp_gemv_name(name: &str) -> bool {
    name.ends_with("mlp.gate_proj.weight")
        || name.ends_with("mlp.up_proj.weight")
        || name.ends_with("mlp.down_proj.weight")
}

/// Walk only MLP GEMV rows of an HQ38M20 catalog. Does not open Metal and
/// does not read full payloads — 64-byte prefixes are enough to classify
/// codecs 0–3. Used to prove a pack admits before a GPU lane loads it.
pub fn assert_mixed_mlp_native_catalog(root: impl AsRef<Path>) -> Result<()> {
    let rows = parse_qwen38_mixed_catalog(root.as_ref())?;
    let mut kinds = HashMap::new();
    for row in &rows {
        if !is_mixed_mlp_gemv_name(&row.name) {
            continue;
        }
        if row.codec > 3 && row.codec != 5 {
            continue;
        }
        let prefix = read_catalog_prefix(row, 64)?;
        let lane = classify_qwen38_mixed_payload(row.codec, &prefix, &row.name, &row.shape)?;
        if let Some(kind) = mixed_mlp_native_kind_from_lane(lane) {
            kinds.insert(row.name.clone(), kind);
        }
    }
    assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied())
}

/// Walk `catalog.hq38m20` without opening Metal. Rice stays packed.
pub fn census_qwen38_mixed_catalog(root: impl AsRef<Path>) -> Result<MixedCatalogCensus> {
    let rows = parse_qwen38_mixed_catalog(root.as_ref())?;
    let mut census = MixedCatalogCensus {
        tensors: rows.len(),
        ..MixedCatalogCensus::default()
    };
    for row in &rows {
        let payload = read_catalog_payload(row)?;
        match classify_qwen38_mixed_payload(row.codec, &payload, &row.name, &row.shape) {
            Ok(MixedCatalogLane::Packed(codec)) => match mixed_gpu_layout(codec, &payload) {
                Ok(_) => match codec {
                    0 => census.binary += 1,
                    1 => census.residual += 1,
                    2 => census.hgravs += 1,
                    3 => census.uniform += 1,
                    _ => {}
                },
                Err(error) => {
                    census.refused += 1;
                    census.refusals.push(format!("{} layout: {error}", row.name));
                }
            },
            Ok(MixedCatalogLane::Hq30Uq4) => match parse_uniform_q4_header(&payload) {
                Ok(_) => census.q4 += 1,
                Err(error) => {
                    census.refused += 1;
                    census.refusals.push(format!("{} q4: {error}", row.name));
                }
            },
            Ok(MixedCatalogLane::F32v2) => {
                census.f32 += 1;
            }
            Ok(MixedCatalogLane::HgravuVector) => {
                census.f32 += 1;
            }
            Ok(MixedCatalogLane::Affine) => match mixed_gpu_layout(5, &payload) {
                Ok(_) => census.affine += 1,
                Err(error) => {
                    census.refused += 1;
                    census.refusals.push(format!("{} affine: {error}", row.name));
                }
            },
            Err(error) => {
                census.refused += 1;
                census.refusals.push(format!("{error}"));
            }
        }
    }
    Ok(census)
}

#[derive(Clone, Debug)]
struct Qwen38MixedCatalogRow {
    name: String,
    codec: u8,
    shape: Vec<usize>,
    segment_path: PathBuf,
    offset: u64,
    nbytes: u64,
}

fn read_u16_at(raw: &[u8], off: usize) -> Result<u16> {
    let slice = raw
        .get(off..off + 2)
        .ok_or_else(|| mixed_error("catalog truncated at u16"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32_at(raw: &[u8], off: usize) -> Result<u32> {
    let slice = raw
        .get(off..off + 4)
        .ok_or_else(|| mixed_error("catalog truncated at u32"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64_at(raw: &[u8], off: usize) -> Result<u64> {
    let slice = raw
        .get(off..off + 8)
        .ok_or_else(|| mixed_error("catalog truncated at u64"))?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

/// Segment filenames are `segments/<name>` by default. An absolute filename
/// (used when a sandbox cannot hardlink into `segments/`) is kept as-is so
/// the catalog can name already-packed blobs without copying 4.34 GB.
fn resolve_mixed_segment_path(root: &Path, filename: &str) -> PathBuf {
    let raw = Path::new(filename);
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        root.join("segments").join(filename)
    }
}

fn parse_qwen38_mixed_catalog(root: &Path) -> Result<Vec<Qwen38MixedCatalogRow>> {
    let catalog_path = root.join(QWEN38_MIXED_CATALOG_NAME);
    let raw = fs::read(&catalog_path).map_err(|error| {
        mixed_error(format!("cannot read {}: {error}", catalog_path.display()))
    })?;
    if raw.len() < 32 || raw[..8] != QWEN38_MIXED_CATALOG_MAGIC {
        return Err(mixed_error("catalog magic is not HQ38M20"));
    }
    let version = read_u32_at(&raw, 8)?;
    if version != QWEN38_MIXED_CATALOG_VERSION {
        return Err(mixed_error(format!("unsupported catalog version {version}")));
    }
    let n_tensors = read_u32_at(&raw, 12)? as usize;
    let n_segments = read_u32_at(&raw, 16)? as usize;
    let name_blob_bytes = read_u32_at(&raw, 24)? as usize;
    let mut cursor = 32usize;
    let mut by_id: HashMap<u16, PathBuf> = HashMap::new();
    for _ in 0..n_segments {
        let id = read_u16_at(&raw, cursor)?;
        let name_len = read_u16_at(&raw, cursor + 2)? as usize;
        cursor += 44;
        let filename = raw
            .get(cursor..cursor + name_len)
            .ok_or_else(|| mixed_error("segment name truncated"))?;
        let filename = std::str::from_utf8(filename)
            .map_err(|_| mixed_error("segment name is not utf-8"))?
            .to_owned();
        cursor += name_len;
        by_id.insert(id, resolve_mixed_segment_path(root, &filename));
    }
    let table_bytes = n_tensors
        .checked_mul(QWEN38_MIXED_RECORD_SIZE)
        .ok_or_else(|| mixed_error("catalog table size overflow"))?;
    let table = raw
        .get(cursor..cursor + table_bytes)
        .ok_or_else(|| mixed_error("catalog tensor table truncated"))?;
    cursor += table_bytes;
    let name_blob = raw
        .get(cursor..cursor + name_blob_bytes)
        .ok_or_else(|| mixed_error("catalog name blob truncated"))?;
    let mut rows = Vec::with_capacity(n_tensors);
    for index in 0..n_tensors {
        let rec = &table[index * QWEN38_MIXED_RECORD_SIZE
            ..(index + 1) * QWEN38_MIXED_RECORD_SIZE];
        let name_off = read_u32_at(rec, 0)? as usize;
        let name_len = read_u16_at(rec, 4)? as usize;
        let codec = rec[6];
        let ndim = rec[8] as usize;
        if ndim > 4 {
            return Err(mixed_error("catalog ndim exceeds 4"));
        }
        let mut shape = Vec::with_capacity(ndim);
        for dim in 0..ndim {
            shape.push(read_u32_at(rec, 12 + dim * 4)? as usize);
        }
        let segment_id = read_u16_at(rec, 36)?;
        let offset = read_u64_at(rec, 40)?;
        let nbytes = read_u64_at(rec, 48)?;
        let name = name_blob
            .get(name_off..name_off + name_len)
            .ok_or_else(|| mixed_error("tensor name out of blob"))?;
        let name = std::str::from_utf8(name)
            .map_err(|_| mixed_error("tensor name is not utf-8"))?
            .to_owned();
        let segment_path = by_id
            .get(&segment_id)
            .cloned()
            .ok_or_else(|| mixed_error(format!("unknown segment_id {segment_id}")))?;
        rows.push(Qwen38MixedCatalogRow {
            name,
            codec,
            shape,
            segment_path,
            offset,
            nbytes,
        });
    }
    Ok(rows)
}

fn read_catalog_payload(row: &Qwen38MixedCatalogRow) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = fs::File::open(&row.segment_path).map_err(|error| {
        mixed_error(format!(
            "cannot open {}: {error}",
            row.segment_path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(row.offset)).map_err(|error| {
        mixed_error(format!("seek {}: {error}", row.name))
    })?;
    let n = usize::try_from(row.nbytes).map_err(|_| mixed_error("payload exceeds usize"))?;
    let mut payload = vec![0u8; n];
    file.read_exact(&mut payload)
        .map_err(|error| mixed_error(format!("read {}: {error}", row.name)))?;
    Ok(payload)
}

fn dequant_hgravu_vector(payload: &[u8], name: &str) -> Result<Vec<f32>> {
    let layout = mixed_gpu_layout(3, payload)?;
    let MixedGpuKind::Uniform(factor) = layout.kind else {
        return Err(mixed_error(format!("{name} codec-3 layout is not uniform")));
    };
    let elements = (factor.rows as usize)
        .checked_mul(factor.cols as usize)
        .ok_or_else(|| mixed_error(format!("{name} uniform length overflows")))?;
    if elements > 65_536 {
        return Err(mixed_error(format!(
            "{name} dequant refuses {elements} elements (dense W)"
        )));
    }
    let scales = &payload[factor.scale_off..factor.scale_off + factor.scale_bytes];
    let mut scales_f16 = Vec::with_capacity(factor.scale_bytes / 2);
    for chunk in scales.chunks_exact(2) {
        scales_f16.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    let packed = UniformFactorPacked {
        rows: factor.rows as usize,
        cols: factor.cols as usize,
        bits: u8::try_from(factor.bits).map_err(|_| mixed_error("HGRAVU01 bits"))?,
        group_size: factor.group_size as usize,
        groups: elements.div_ceil(factor.group_size as usize),
        bound: u16::try_from(factor.bound).map_err(|_| mixed_error("HGRAVU01 bound"))?,
        scales_f16,
        codes: payload[factor.code_off..factor.code_off + factor.code_bytes].to_vec(),
    };
    let mut values = Vec::with_capacity(elements);
    for row in 0..packed.rows {
        for col in 0..packed.cols {
            values.push(uniform_factor_value(&packed, row, col));
        }
    }
    mlx_residual_norm_to_delta_named(name, &mut values);
    Ok(values)
}

pub const QWEN38_Q4_MATVEC_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_Q4_GROUP128_MATVEC_KERNEL: &str =
    "qwen_uniform_q4_group128_matvec_geo_tpr64_tg128";
pub const QWEN38_Q4_ADDR_PROBE_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe";
pub const QWEN38_Q4_DECODE_PROBE_KERNEL: &str =
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe";
pub const QWEN38_F32_STREAM_PROBE_KERNEL: &str = "qwen38_f32_stream_probe";
pub const QWEN38_HGRAVU01_Q3_GEO_TPR64: &str =
    "qwen_uniform_q3_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_HGRAVU01_Q3_G128_GEO_TPR64: &str =
    "qwen_uniform_q3_group128_matvec_geo_tpr64_tg128";
pub const QWEN38_HGRAVU01_Q4_GEO_TPR64: &str =
    "qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_AFFINE_Q2_SERIAL: &str = "qwen_affine_q2_group32_matvec";
pub const QWEN38_AFFINE_Q2_GEO_TPR64: &str = "qwen_affine_q2_group32_matvec_geo_tpr64_tg128";
pub const QWEN38_Q2F_SERIAL: &str = "qwen_q2f_group64_matvec";
pub const QWEN38_Q2F_GEO_TPR64: &str = "qwen_q2f_group64_matvec_geo_tpr64_tg128";
pub const QWEN38_Q2F_QKV_GEO_KERNEL: &str =
    "qwen_q2f_group64_matvec_qkv_geo_tpr64_tg128";
pub const QWEN38_Q2F_PAIR_GEO_KERNEL: &str =
    "qwen_q2f_group64_matvec_pair_geo_tpr64_tg128";
pub const QWEN38_Q2F_GATE_UP_KERNEL: &str = "qwen_q2f_group64_matvec_gate_up_geo_tpr64_tg128";
pub const QWEN38_Q2F_GATE_UP_SWIGLU_KERNEL: &str =
    "qwen_q2f_group64_matvec_gate_up_swiglu_geo_tpr64_tg128";
pub const QWEN38_Q2F_PIPE: &str = "qwen_q2f_group64_matvec_pipe_tpr64_tg128";
pub const QWEN38_Q2F_SPLITK4: &str = "qwen_q2f_group64_matvec_splitk4_tg256";
pub const QWEN38_Q2F_SPLITK4_VEC: &str = "qwen_q2f_group64_matvec_splitk4_vec_tg256";
pub const QWEN38_Q2F_GATE_UP_PIPE: &str =
    "qwen_q2f_group64_matvec_gate_up_pipe_tpr64_tg128";
pub const QWEN38_Q2F_GATE_UP_SWIGLU_PIPE: &str =
    "qwen_q2f_group64_matvec_gate_up_swiglu_pipe_tpr64_tg128";
pub const QWEN38_Q2F_GATE_UP_SPLITK4: &str =
    "qwen_q2f_group64_matvec_gate_up_splitk4_tg256";
pub const QWEN38_Q2F_GATE_UP_SWIGLU_SPLITK4: &str =
    "qwen_q2f_group64_matvec_gate_up_swiglu_splitk4_tg256";
pub const QWEN38_Q2F_GATE_UP_SPLITK4_VEC: &str =
    "qwen_q2f_group64_matvec_gate_up_splitk4_vec_tg256";
pub const QWEN38_Q2F_GATE_UP_SWIGLU_SPLITK4_VEC: &str =
    "qwen_q2f_group64_matvec_gate_up_swiglu_splitk4_vec_tg256";
pub const QWEN38_HGRAFV_EMBED: &str = "qwen38_hgrafv_embedding_lookup";

fn qwen38_q2f_matvec_launch(
    geo: Affine2Geo,
    rows: u32,
) -> (&'static str, (u32, u32, u32), (u32, u32, u32)) {
    let tg = match geo {
        Affine2Geo::SplitK4 | Affine2Geo::SplitK4Vec => 256u32,
        _ => 128u32,
    };
    let name = match geo {
        Affine2Geo::Pipe => QWEN38_Q2F_PIPE,
        Affine2Geo::SplitK4 => QWEN38_Q2F_SPLITK4,
        Affine2Geo::SplitK4Vec => QWEN38_Q2F_SPLITK4_VEC,
        _ => QWEN38_Q2F_GEO_TPR64,
    };
    let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
    (name, (grid, 1, 1), (tg, 1, 1))
}

fn qwen38_q2f_gate_up_launch(
    geo: Affine2Geo,
    with_swiglu: bool,
    rows: u32,
) -> (&'static str, (u32, u32, u32), (u32, u32, u32)) {
    let tg = match geo {
        Affine2Geo::SplitK4 | Affine2Geo::SplitK4Vec => 256u32,
        _ => 128u32,
    };
    let name = match (geo, with_swiglu) {
        (Affine2Geo::Pipe, false) => QWEN38_Q2F_GATE_UP_PIPE,
        (Affine2Geo::Pipe, true) => QWEN38_Q2F_GATE_UP_SWIGLU_PIPE,
        (Affine2Geo::SplitK4, false) => QWEN38_Q2F_GATE_UP_SPLITK4,
        (Affine2Geo::SplitK4, true) => QWEN38_Q2F_GATE_UP_SWIGLU_SPLITK4,
        (Affine2Geo::SplitK4Vec, false) => QWEN38_Q2F_GATE_UP_SPLITK4_VEC,
        (Affine2Geo::SplitK4Vec, true) => QWEN38_Q2F_GATE_UP_SWIGLU_SPLITK4_VEC,
        (_, false) => QWEN38_Q2F_GATE_UP_KERNEL,
        (_, true) => QWEN38_Q2F_GATE_UP_SWIGLU_KERNEL,
    };
    let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
    (name, (grid, 1, 1), (tg, 1, 1))
}

/// G0-class launch for Affine HGRAVF01 q2 at group 32 or 64.
/// None selects the serial `qwen_affine_q2_group32_matvec`.
/// Never falls through to HGRAVU01. Kernel names keep the group32 family.
pub fn qwen38_affine_q2_geo_tpr64_launch(
    group_size: u32,
    rows: u32,
    cols: u32,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    qwen38_affine_q2_launch(Affine2Geo::Tpr64, group_size, rows, cols)
}

/// Launch for a named affine2 geometry. g64-specialized levers refuse group 32.
pub fn qwen38_affine_q2_launch(
    geo: Affine2Geo,
    group_size: u32,
    rows: u32,
    cols: u32,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    qwen38_affine_q2_launch_with_recon_fuse(
        geo,
        group_size,
        rows,
        cols,
        qwen38_recon_fuse_enabled(),
    )
}

fn qwen38_affine_q2_launch_with_recon_fuse(
    geo: Affine2Geo,
    group_size: u32,
    rows: u32,
    cols: u32,
    recon_fuse: bool,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    if !recon_fuse
        || !affine_group_size_supported(group_size as usize)
        || cols % group_size != 0
    {
        return None;
    }
    if geo.is_g64_specialized() && group_size != 64 {
        return None;
    }
    match geo {
        Affine2Geo::Tpr64 => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_GEO_TPR64, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::RuntimeDiv => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_GEO_TPR64_RUNTIME_DIV, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::QmvFast => {
            let tg = 64u32;
            let grid = rows.div_ceil(8).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_QMVFAST, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::Wide64 => {
            let tg = 128u32;
            let grid = rows.div_ceil(4).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_WIDE64, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::Tgx => {
            let tg = 256u32;
            let grid = rows.div_ceil(8).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_TGX, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::Tgsb => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_TGSB, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::Pipe => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_PIPE, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::SplitK4 => {
            let tg = 256u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_SPLITK4, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::SplitK4Vec => {
            let tg = 256u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_SPLITK4_VEC, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::AccFuse => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_ACCFUSE, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::FoldAddqx => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_FOLD_ADDQX, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::Bitcast => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_BITCAST, (grid, 1, 1), (tg, 1, 1)))
        }
        Affine2Geo::BiasPrep | Affine2Geo::BiasPrepDrop => {
            // mlp_down and other single GEMVs stay on tpr64. BiasPrep is
            // a fused gate_up_swiglu organ cut (N031 owns down).
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            Some((QWEN38_AFFINE_Q2_GEO_TPR64, (grid, 1, 1), (tg, 1, 1)))
        }
    }
}

fn qwen38_affine_gate_up_launch(
    geo: Affine2Geo,
    with_swiglu: bool,
    rows: u32,
) -> (&'static str, (u32, u32, u32), (u32, u32, u32)) {
    match geo {
        Affine2Geo::QmvFast => {
            let tg = 64u32;
            let grid = rows.div_ceil(8).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_QMVFAST
            } else {
                QWEN38_AFFINE_GATE_UP_QMVFAST
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::Wide64 => {
            let tg = 128u32;
            let grid = rows.div_ceil(4).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_WIDE64
            } else {
                QWEN38_AFFINE_GATE_UP_WIDE64
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::Tgx => {
            let tg = 256u32;
            let grid = rows.div_ceil(8).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_TGX
            } else {
                QWEN38_AFFINE_GATE_UP_TGX
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::Tgsb => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_TGSB
            } else {
                QWEN38_AFFINE_GATE_UP_TGSB
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::Pipe => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_PIPE
            } else {
                QWEN38_AFFINE_GATE_UP_PIPE
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::SplitK4 => {
            let tg = 256u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_SPLITK4
            } else {
                QWEN38_AFFINE_GATE_UP_SPLITK4
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::SplitK4Vec => {
            let tg = 256u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = match with_swiglu {
                true => QWEN38_AFFINE_GATE_UP_SWIGLU_SPLITK4_VEC,
                false => QWEN38_AFFINE_GATE_UP_SPLITK4_VEC,
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::AccFuse => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_ACCFUSE
            } else {
                QWEN38_AFFINE_GATE_UP_ACCFUSE
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::FoldAddqx => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX
            } else {
                QWEN38_AFFINE_GATE_UP_FOLD_ADDQX
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::Bitcast => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            // Only the swiglu-fused form is written, because that is the one the
            // resident dispatches. The unfused form falls back to production
            // rather than naming a kernel that does not exist.
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_BITCAST
            } else {
                QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::BiasPrep => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_BIASPREP
            } else {
                QWEN38_AFFINE_GATE_UP_BIASPREP
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::BiasPrepDrop => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_BIASPREP_DROP
            } else {
                QWEN38_AFFINE_GATE_UP_BIASPREP
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
        Affine2Geo::Tpr64 | Affine2Geo::RuntimeDiv => {
            let tg = 128u32;
            let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
            let name = if with_swiglu {
                QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL
            } else {
                QWEN38_AFFINE_GATE_UP_KERNEL
            };
            (name, (grid, 1, 1), (tg, 1, 1))
        }
    }
}

/// G0-class launch for Uniform HGRAVU01 bits 3/4. None leaves the
/// incumbent simd / simd3 / uniform8 / serial path in `dispatch_factor`.
/// HGRAVS r160 factors stay on that path because they are not Uniform.
///
/// bits=3 group=64  → qwen_uniform_q3_group64_matvec_geo_tpr64_tg128
/// bits=3 group=128 → qwen_uniform_q3_group128_matvec_geo_tpr64_tg128
/// bits=4 group=64  → qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128
pub fn qwen38_hgravu01_geo_tpr64_launch(
    bits: u32,
    group_size: u32,
    rows: u32,
    cols: u32,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    qwen38_hgravu01_geo_tpr64_launch_with_recon_fuse(
        bits,
        group_size,
        rows,
        cols,
        qwen38_recon_fuse_enabled(),
    )
}

fn qwen38_hgravu01_geo_tpr64_launch_with_recon_fuse(
    bits: u32,
    group_size: u32,
    rows: u32,
    cols: u32,
    recon_fuse: bool,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    if !recon_fuse || group_size == 0 || cols % group_size != 0 {
        return None;
    }
    let name = match (bits, group_size) {
        (3, 64) => QWEN38_HGRAVU01_Q3_GEO_TPR64,
        (3, 128) => QWEN38_HGRAVU01_Q3_G128_GEO_TPR64,
        (4, 64) => QWEN38_HGRAVU01_Q4_GEO_TPR64,
        _ => return None,
    };
    let tg = 128u32;
    let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
    Some((name, (grid, 1, 1), (tg, 1, 1)))
}

/// HQ30UQ4 geo_tpr64 bind. Supported group sizes are exactly 64 and 128.
/// Unsupported sizes, or a width that is not a multiple of the group,
/// return None so the caller can refuse rather than silently fall back.
pub fn qwen38_uniform_q4_geo_tpr64_launch(
    group_size: u32,
    rows: u32,
    cols: u32,
) -> Option<(&'static str, (u32, u32, u32), (u32, u32, u32))> {
    if !uniform_q4_group_size_supported(group_size as usize) || cols % group_size != 0 {
        return None;
    }
    let name = match group_size {
        64 => qwen38_q4_kernel(QWEN38_Q4_MATVEC_KERNEL),
        128 => QWEN38_Q4_GROUP128_MATVEC_KERNEL,
        _ => return None,
    };
    let tg = 128u32;
    let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
    Some((name, (grid, 1, 1), (tg, 1, 1)))
}

/// Shipped uniform-Q4 matvec bindings. The Qwen3.8 default is the geometry-
/// sweep winner (`geo_tpr64_tg128`), tuned on Q80's 512×2048 organs. The
/// other names are already in `qwen_uniform_q4.metal`; this enum only
/// retargets launch geometry. It does not generate new shaders.
pub const QWEN38_Q4_GEO_ENV: &str = "HAWKING_QWEN38_Q4_GEO";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38MatvecKernel {
    GeoTpr64Tg128,
    Vecgroup,
    VecgroupX64,
    VecgroupR4,
}

impl Qwen38MatvecKernel {
    /// Parse the explicit uniform-Q4 geometry selector.
    ///
    /// This selector is intentionally strict. A misspelled protected-run
    /// mutation must fail before a token is measured instead of silently
    /// recording the incumbent geometry under the candidate's name.
    pub fn from_value(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "geo" | "tpr64" | "geo_tpr64" | "geo_tpr64_tg128" => Self::GeoTpr64Tg128,
            "vecgroup" | "simdgroup" => Self::Vecgroup,
            "vecgroup_x64" | "x64" => Self::VecgroupX64,
            "vecgroup_r4" | "r4" => Self::VecgroupR4,
            other => panic!(
                "{QWEN38_Q4_GEO_ENV}={other:?} is not a recognised value; use geo | vecgroup | vecgroup_x64 | vecgroup_r4"
            ),
        }
    }

    /// Resolve the standalone uniform-Q4 geometry once when the resident
    /// session attaches. Unset preserves the shipped geo_tpr64 control.
    pub fn from_env() -> Self {
        std::env::var(QWEN38_Q4_GEO_ENV)
            .map(|value| Self::from_value(&value))
            .unwrap_or(Self::GeoTpr64Tg128)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GeoTpr64Tg128 => QWEN38_Q4_MATVEC_KERNEL,
            Self::Vecgroup => "qwen_uniform_q4_group64_matvec_vecgroup",
            Self::VecgroupX64 => "qwen_uniform_q4_group64_matvec_vecgroup_x64",
            Self::VecgroupR4 => "qwen_uniform_q4_group64_matvec_vecgroup_r4",
        }
    }

    /// (grid, threadgroup) for `rows` output elements.
    pub fn launch(self, rows: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
        match self {
            Self::GeoTpr64Tg128 => {
                let tg = 128u32;
                let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::Vecgroup => {
                let tg = 256u32;
                let grid = rows.div_ceil(8).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::VecgroupX64 => {
                let tg = 256u32;
                let grid = rows.div_ceil(4).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
            Self::VecgroupR4 => {
                let tg = 256u32;
                let grid = rows.div_ceil(32).saturating_mul(tg).max(tg);
                ((grid, 1, 1), (tg, 1, 1))
            }
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::GeoTpr64Tg128,
            Self::Vecgroup,
            Self::VecgroupX64,
            Self::VecgroupR4,
        ]
    }
}

/// Per-class GPU times from residual-correct split command buffers.
/// Each field's `gpu_ns` is `GPUEndTime-GPUStartTime` on that class's CBs.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Qwen38ClassTiming {
    pub embed_gpu_ns: Option<u64>,
    pub embed_wait_ns: u64,
    pub mixer_gpu_ns: u64,
    pub mixer_wait_ns: u64,
    pub mlp_gpu_ns: u64,
    pub mlp_wait_ns: u64,
    pub terminal_gpu_ns: Option<u64>,
    pub terminal_wait_ns: u64,
    pub deltanet_gpu_ns: u64,
    pub gqa_gpu_ns: u64,
    pub sampled: u32,
    pub layer_mlp_gpu_ns: Vec<u64>,
    pub layer_mixer_gpu_ns: Vec<u64>,
}

pub fn render_qwen38_user_chat(user_text: &str) -> String {
    format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
}

/// Per-session Metal workspace size. Independent of the weight set.
/// KV grows with `max_seq_len`; DeltaNet state does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Qwen38WorkspaceBytes {
    pub max_seq_len: usize,
    pub activation_bytes: usize,
    pub deltanet_state_bytes: usize,
    pub gqa_kv_bytes: usize,
    pub total_bytes: usize,
}

pub fn qwen38_workspace_bytes(max_seq_len: usize) -> Result<Qwen38WorkspaceBytes> {
    if max_seq_len == 0 {
        return Err(Error::Model("qwen38 max_seq_len must be positive".into()));
    }
    let layout = Qwen38DeltaNetLayout::source_exact();
    let f32b = |n: usize| {
        n.checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))
    };
    let hidden = f32b(QWEN38_HIDDEN)?;
    let qkvz = f32b(layout.qkvz_rows())?;
    let ba = f32b(layout.ba_rows())?;
    let value = f32b(layout.value_elements())?;
    let q_proj = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)?;
    let kv = f32b(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
    let query = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)?;
    let mid = f32b(QWEN38_INTERMEDIATE)?;
    let logits = f32b(QWEN38_VOCAB)?;
    let conv = f32b(48 * layout.conv_state_elements())?;
    let rec = f32b(48 * layout.recurrent_state_elements())?;
    let kv_cache = f32b(
        QWEN38_GQA_LAYERS
            .checked_mul(max_seq_len)
            .and_then(|n| n.checked_mul(QWEN38_GQA_KV_HEADS))
            .and_then(|n| n.checked_mul(QWEN38_GQA_HEAD_DIM))
            .ok_or_else(|| Error::Model("qwen38 KV cache overflow".into()))?,
    )?;
    let hgravs = f32b(QWEN38_MIXED_HGRAVS_RANK)?;
    let split_qkv = f32b(crate::model::qwen38_geometry::QWEN38_IN_PROJ_QKV_ROWS)?;
    let split_b = f32b(crate::model::qwen38_geometry::QWEN38_IN_PROJ_B_ROWS)?;
    let split_a = f32b(crate::model::qwen38_geometry::QWEN38_IN_PROJ_A_ROWS)?;
    let sampled = std::mem::size_of::<u32>();
    let heads_f32 = f32b(layout.value_heads)?;
    let xsum64 = f32b(QWEN38_XSUM64_CAP)?;
    let activation = hidden
        .checked_mul(2)
        .and_then(|n| n.checked_add(qkvz))
        .and_then(|n| n.checked_add(ba))
        .and_then(|n| n.checked_add(value.checked_mul(6)?))
        .and_then(|n| n.checked_add(heads_f32.checked_mul(2)?))
        .and_then(|n| n.checked_add(hidden.checked_mul(2)?))
        .and_then(|n| n.checked_add(q_proj))
        .and_then(|n| n.checked_add(kv.checked_mul(2)?))
        .and_then(|n| n.checked_add(query.checked_mul(3)?))
        .and_then(|n| n.checked_add(mid.checked_mul(3)?))
        .and_then(|n| n.checked_add(hidden))
        .and_then(|n| n.checked_add(logits))
        .and_then(|n| n.checked_add(sampled))
        .and_then(|n| n.checked_add(hgravs))
        .and_then(|n| n.checked_add(split_qkv))
        .and_then(|n| n.checked_add(split_b))
        .and_then(|n| n.checked_add(split_a))
        .and_then(|n| n.checked_add(xsum64))
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    let deltanet = conv
        .checked_add(rec)
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    let gqa = kv_cache
        .checked_mul(2)
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    let total = activation
        .checked_add(deltanet)
        .and_then(|n| n.checked_add(gqa))
        .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))?;
    Ok(Qwen38WorkspaceBytes {
        max_seq_len,
        activation_bytes: activation,
        deltanet_state_bytes: deltanet,
        gqa_kv_bytes: gqa,
        total_bytes: total,
    })
}

pub fn load_qwen38_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
}

#[cfg(target_os = "macos")]
mod device {
    use super::*;
    use crate::json_constrain::{argmax_f32_metal_tiebreak, JsonConstraint, JsonVocabIndex};
    use crate::kernels::{
        mha_decode_f32_tcb, qwen_next_add_residual_tcb, sample_argmax_f32_tcb,
    };
    use crate::metal::{CommandBufferTiming, MetalContext, PinnedBuffer, TokenCommandBuffer};
    use std::cell::Cell;
    use std::thread;
    use std::time::Instant;

    fn zero_buffer(buffer: &PinnedBuffer) {
        let len = buffer.length() as usize;
        unsafe {
            std::ptr::write_bytes(buffer.contents() as *mut u8, 0, len);
        }
    }

    struct Q4Weight {
        rows: usize,
        cols: usize,
        group_size: usize,
        /// Packed-Q4 ABI value reused by every token dispatch. Keep the
        /// historical fallback divisor for unsupported headers; the dispatch
        /// selector remains the authority for refusing unsupported geometry.
        groups_per_row: u32,
        codes: PinnedBuffer,
        scales: PinnedBuffer,
    }

    struct GpuBinary {
        signs: PinnedBuffer,
        scales: PinnedBuffer,
        rows: u32,
        cols: u32,
        group_size: u32,
        groups_per_row: u32,
    }

    struct GpuResidual {
        binary: GpuBinary,
        indices: PinnedBuffer,
        row_ptr: PinnedBuffer,
        residual_signs: PinnedBuffer,
        residual_scale_f16: u32,
    }

    struct GpuHgravs {
        left_codes: PinnedBuffer,
        left_scales: PinnedBuffer,
        right_codes: PinnedBuffer,
        right_scales: PinnedBuffer,
        left_rows: u32,
        left_cols: u32,
        right_rows: u32,
        right_cols: u32,
        group_size: u32,
        bits: u32,
        bound: u32,
    }

    struct GpuUniform {
        codes: PinnedBuffer,
        scales: PinnedBuffer,
        rows: u32,
        cols: u32,
        group_size: u32,
        bits: u32,
        bound: u32,
    }

    struct GpuAffine {
        codes: PinnedBuffer,
        scales: PinnedBuffer,
        /// None = Q2F (w = (q-1.5)*delta). Some = affine2 (w = q*scale+bias).
        biases: Option<PinnedBuffer>,
        rows: u32,
        cols: u32,
        group_size: u32,
        bits: u32,
    }

    enum MixedGpuWeight {
        Binary(GpuBinary),
        Residual(GpuResidual),
        Hgravs(GpuHgravs),
        Uniform(GpuUniform),
        Affine(GpuAffine),
    }

    impl MixedGpuWeight {
        fn resident_bytes(&self) -> u64 {
            match self {
                Self::Binary(body) => body.signs.length() + body.scales.length(),
                Self::Residual(body) => {
                    body.binary.signs.length()
                        + body.binary.scales.length()
                        + body.indices.length()
                        + body.row_ptr.length()
                        + body.residual_signs.length()
                }
                Self::Hgravs(body) => {
                    body.left_codes.length()
                        + body.left_scales.length()
                        + body.right_codes.length()
                        + body.right_scales.length()
                }
                Self::Uniform(body) => body.codes.length() + body.scales.length(),
                Self::Affine(body) => {
                    body.codes.length()
                        + body.scales.length()
                        + body.biases.as_ref().map(|b| b.length()).unwrap_or(0)
                }
            }
        }
    }

    /// One resident Metal copy of the Qwen3.8 catalog. Sessions clone the
    /// `Arc` and allocate only workspace / KV.
    pub struct Qwen38HybridWeights {
        context: MetalContext,
        q4: HashMap<String, Q4Weight>,
        f32s: HashMap<String, PinnedBuffer>,
        mixed: HashMap<String, MixedGpuWeight>,
        /// GEMV weights reconstructed to dense float. Stays 0 on the packed path.
        pub dense_w_materialized: u64,
    }

    impl Qwen38HybridWeights {
        pub fn load(root: impl AsRef<Path>) -> Result<Self> {
            qwen38_assert_schedule_intact()?;
            let root = root.as_ref();
            if root.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
                return Self::load_mixed(root);
            }
            let (_manifest, rows) = load_qwen38_manifest(root)?;
            if rows.len() != QWEN38_EXPECTED_CATALOG_TENSORS {
                return Err(Error::Model(format!(
                    "qwen38 catalog has {} tensors, expected {QWEN38_EXPECTED_CATALOG_TENSORS}",
                    rows.len()
                )));
            }
            eprintln!(
                "qwen38-decode opening Metal + {} catalog tensors",
                rows.len()
            );
            let context = MetalContext::new_with_trace(qwen38_trace_dispatch_enabled())?;
            let mut q4 = HashMap::new();
            let mut f32s = HashMap::new();
            let tensors_dir = root.join("tensors");
            for (i, row) in rows.iter().enumerate() {
                if i % 50 == 0 {
                    eprintln!("qwen38-decode upload {i}/{}", rows.len());
                }
                let path = tensors_dir.join(&row.artifact);
                let payload = fs::read(&path).map_err(|error| {
                    Error::Model(format!("cannot read {}: {error}", path.display()))
                })?;
                match row.kind.as_str() {
                    "q4" => {
                        let header = parse_uniform_q4_header(&payload)?;
                        let scales = &payload[header.scale_offset..header.sign_offset];
                        let codes = &payload[header.sign_offset..header.payload_bytes];
                        let (rows_n, cols) = match header.shape.as_slice() {
                            [r, c] => (*r, *c),
                            other => {
                                return Err(Error::Model(format!(
                                    "{} Q4 rank {:?} is not a matrix",
                                    row.name, other
                                )))
                            }
                        };
                        let groups_per_row = if header.group_size == UNIFORM_Q4_GROUP_SIZE_128 {
                            cols.div_ceil(UNIFORM_Q4_GROUP_SIZE_128) as u32
                        } else {
                            cols.div_ceil(UNIFORM_Q4_GROUP_SIZE) as u32
                        };
                        q4.insert(
                            row.name.clone(),
                            Q4Weight {
                                rows: rows_n,
                                cols,
                                group_size: header.group_size,
                                groups_per_row,
                                codes: context.new_buffer_with_bytes_checked(codes)?,
                                scales: context.new_buffer_with_bytes_checked(scales)?,
                            },
                        );
                    }
                    "f32" => {
                        let values = read_qwen38_f32_payload(&payload)?;
                        f32s.insert(
                            row.name.clone(),
                            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(&values))?,
                        );
                    }
                    other => {
                        return Err(Error::Model(format!(
                            "qwen38 catalog kind {other:?} is not q4/f32"
                        )))
                    }
                }
            }
            Ok(Self {
                context,
                q4,
                f32s,
                mixed: HashMap::new(),
                dense_w_materialized: 0,
            })
        }

        fn load_mixed(root: &Path) -> Result<Self> {
            let rows = parse_qwen38_mixed_catalog(root)?;
            if rows.is_empty() {
                return Err(mixed_error("HQ38M20 catalog has no tensors"));
            }
            eprintln!(
                "qwen38-decode opening mixed HQ38M20 + {} catalog tensors (no reconstruct-to-Q4)",
                rows.len()
            );
            let context = MetalContext::new_with_trace(qwen38_trace_dispatch_enabled())?;
            let mut q4 = HashMap::new();
            let mut f32s = HashMap::new();
            let mut mixed = HashMap::new();
            let mut census = MixedCatalogCensus {
                tensors: rows.len(),
                ..MixedCatalogCensus::default()
            };
            for (i, row) in rows.iter().enumerate() {
                if i % 50 == 0 {
                    eprintln!("qwen38-decode mixed upload {i}/{}", rows.len());
                }
                let payload = read_catalog_payload(row)?;
                match classify_qwen38_mixed_payload(
                    row.codec,
                    &payload,
                    &row.name,
                    &row.shape,
                )? {
                    MixedCatalogLane::Packed(codec) => {
                        mixed.insert(
                            row.name.clone(),
                            Qwen38HybridDecodeSession::upload_mixed(
                                &context, codec, &payload, &row.name,
                            )?,
                        );
                        match codec {
                            0 => census.binary += 1,
                            1 => census.residual += 1,
                            2 => census.hgravs += 1,
                            3 => census.uniform += 1,
                            _ => {}
                        }
                    }
                    MixedCatalogLane::Hq30Uq4 => {
                        let header = parse_uniform_q4_header(&payload)?;
                        let scales = &payload[header.scale_offset..header.sign_offset];
                        let codes = &payload[header.sign_offset..header.payload_bytes];
                        let (rows_n, cols) = match header.shape.as_slice() {
                            [r, c] => (*r, *c),
                            other => {
                                return Err(mixed_error(format!(
                                    "{} HQ30UQ4 rank {:?} is not a matrix",
                                    row.name, other
                                )))
                            }
                        };
                        let groups_per_row = if header.group_size == UNIFORM_Q4_GROUP_SIZE_128 {
                            cols.div_ceil(UNIFORM_Q4_GROUP_SIZE_128) as u32
                        } else {
                            cols.div_ceil(UNIFORM_Q4_GROUP_SIZE) as u32
                        };
                        q4.insert(
                            row.name.clone(),
                            Q4Weight {
                                rows: rows_n,
                                cols,
                                group_size: header.group_size,
                                groups_per_row,
                                codes: context.new_buffer_with_bytes_checked(codes)?,
                                scales: context.new_buffer_with_bytes_checked(scales)?,
                            },
                        );
                        census.q4 += 1;
                    }
                    MixedCatalogLane::F32v2 => {
                        // Already HF δ (f32v2 oracle). Do not mlx-delta.
                        let values = read_qwen38_f32_payload(&payload)?;
                        f32s.insert(
                            row.name.clone(),
                            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(
                                &values,
                            ))?,
                        );
                        census.f32 += 1;
                    }
                    MixedCatalogLane::HgravuVector => {
                        let values = dequant_hgravu_vector(&payload, &row.name)?;
                        f32s.insert(
                            row.name.clone(),
                            context.new_buffer_with_bytes_checked(bytemuck::cast_slice(
                                &values,
                            ))?,
                        );
                        census.f32 += 1;
                    }
                    MixedCatalogLane::Affine => {
                        let layout = mixed_gpu_layout(5, &payload)?;
                        let MixedGpuKind::Affine {
                            scale_off,
                            scale_bytes,
                            bias_off,
                            bias_bytes,
                            code_off,
                            code_bytes,
                            group_size,
                            bits,
                        } = layout.kind
                        else {
                            return Err(mixed_error(format!(
                                "{} codec 5 layout is not Affine",
                                row.name
                            )));
                        };
                        let biases = if bias_bytes == 0 {
                            if std::env::var("HAWKING_Q2F_REUSE_AFFINE2")
                                .map(|v| v != "0")
                                .unwrap_or(false)
                            {
                                let scales = &payload[scale_off..scale_off + scale_bytes];
                                let mut derived = vec![0u8; scale_bytes];
                                for (i, chunk) in scales.chunks_exact(2).enumerate() {
                                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                    let delta = half::f16::from_bits(bits).to_f32();
                                    let bias = half::f16::from_f32(-1.5 * delta).to_bits();
                                    derived[i * 2..i * 2 + 2]
                                        .copy_from_slice(&bias.to_le_bytes());
                                }
                                Some(context.new_buffer_with_bytes_checked(&derived)?)
                            } else {
                                None
                            }
                        } else {
                            Some(context.new_buffer_with_bytes_checked(
                                &payload[bias_off..bias_off + bias_bytes],
                            )?)
                        };
                        mixed.insert(
                            row.name.clone(),
                            MixedGpuWeight::Affine(GpuAffine {
                                codes: context.new_buffer_with_bytes_checked(
                                    &payload[code_off..code_off + code_bytes],
                                )?,
                                scales: context.new_buffer_with_bytes_checked(
                                    &payload[scale_off..scale_off + scale_bytes],
                                )?,
                                biases,
                                rows: layout.rows,
                                cols: layout.cols,
                                group_size,
                                bits,
                            }),
                        );
                        census.affine += 1;
                    }
                }
            }
            eprintln!(
                "qwen38-decode mixed census: tensors={} binary={} residual={} \
                 hgravs={} uniform={} affine={} q4={} f32={} refused={} expanded_to_q4={} \
                 expanded_to_float_gemv={} dense_w_materialized={}",
                census.tensors,
                census.binary,
                census.residual,
                census.hgravs,
                census.uniform,
                census.affine,
                census.q4,
                census.f32,
                census.refused,
                census.expanded_to_q4,
                census.expanded_to_float_gemv,
                census.dense_w_materialized
            );
            eprintln!("{}", qwen38_mixed_k_complete_bind_message());
            {
                let mut n_q2f = 0u32;
                let mut n_q3_g64 = 0u32;
                let mut n_q3_g128 = 0u32;
                let mut n_embed_q3 = 0u32;
                for (name, weight) in &mixed {
                    match weight {
                        MixedGpuWeight::Affine(body) if body.biases.is_none() => n_q2f += 1,
                        MixedGpuWeight::Uniform(body)
                            if body.bits == 3 && body.group_size == 64 =>
                        {
                            n_q3_g64 += 1;
                        }
                        MixedGpuWeight::Uniform(body)
                            if body.bits == 3 && body.group_size == 128 =>
                        {
                            n_q3_g128 += 1;
                            if name.ends_with("embed_tokens.weight") {
                                n_embed_q3 += 1;
                            }
                        }
                        _ => {}
                    }
                }
                let mlp_kernel = if qwen38_recon_fuse_enabled() {
                    qwen38_q2f_matvec_launch(qwen38_q2f_geo_from_env(), 2).0
                } else {
                    QWEN38_Q2F_SERIAL
                };
                eprintln!(
                    "qwen38-decode mixed genome: q2f={n_q2f} q3_g64={n_q3_g64} \
                     q3_g128={n_q3_g128} embed_q3={n_embed_q3} \
                     mlp_kernel={} dn_kernel={} gqa_kernel={} embed_kernel={} \
                     dense_w_materialized={}",
                    mlp_kernel,
                    QWEN38_HGRAVU01_Q3_GEO_TPR64,
                    QWEN38_HGRAVU01_Q3_G128_GEO_TPR64,
                    "qwen38_hgravu_embedding_lookup",
                    census.dense_w_materialized
                );
            }
            if census.affine > 0 {
                let sample = mixed.values().find_map(|weight| match weight {
                    MixedGpuWeight::Affine(body) => Some(body),
                    _ => None,
                });
                let group = sample.map(|b| b.group_size).unwrap_or(0);
                let q2f = sample.map(|b| b.biases.is_none()).unwrap_or(false);
                let kernel = if q2f {
                    if qwen38_recon_fuse_enabled() {
                        qwen38_q2f_matvec_launch(qwen38_q2f_geo_from_env(), 2).0
                    } else {
                        QWEN38_Q2F_SERIAL
                    }
                } else if qwen38_recon_fuse_enabled() {
                    qwen38_affine_q2_launch(
                        Affine2Geo::from_env(),
                        group,
                        2,
                        group.max(1),
                    )
                    .map(|launch| launch.0)
                    .unwrap_or(QWEN38_AFFINE_Q2_SERIAL)
                } else {
                    QWEN38_AFFINE_Q2_SERIAL
                };
                if q2f {
                    eprintln!(
                        "qwen38-decode mixed bind: HGRAVF01 q2f {kernel} group={group} \
                         (delta only, w=(q-1.5)*delta, 4 codes/byte)"
                    );
                } else {
                    eprintln!(
                        "qwen38-decode mixed bind: HGRAVF01 affine2 {kernel} group={group} \
                         (scale+bias, 4 codes/byte)"
                    );
                }
            }
            Qwen38HybridDecodeSession::assert_mixed_mlp_native(&mixed)?;
            Ok(Self {
                context,
                q4,
                f32s,
                mixed,
                dense_w_materialized: census.dense_w_materialized as u64,
            })
        }

        pub fn resident_bytes(&self) -> u64 {
            let q4: u64 = self
                .q4
                .values()
                .map(|w| w.codes.length() + w.scales.length())
                .sum();
            let f32s: u64 = self.f32s.values().map(|b| b.length()).sum();
            let mixed: u64 = self.mixed.values().map(MixedGpuWeight::resident_bytes).sum();
            q4 + f32s + mixed
        }

        fn residency_allocations(&self) -> Vec<&PinnedBuffer> {
            let mut v = Vec::new();
            for w in self.q4.values() {
                v.push(&w.codes);
                v.push(&w.scales);
            }
            for b in self.f32s.values() {
                v.push(b);
            }
            for m in self.mixed.values() {
                match m {
                    MixedGpuWeight::Binary(body) => {
                        v.push(&body.signs);
                        v.push(&body.scales);
                    }
                    MixedGpuWeight::Residual(body) => {
                        v.push(&body.binary.signs);
                        v.push(&body.binary.scales);
                        v.push(&body.indices);
                        v.push(&body.row_ptr);
                        v.push(&body.residual_signs);
                    }
                    MixedGpuWeight::Hgravs(body) => {
                        v.push(&body.left_codes);
                        v.push(&body.left_scales);
                        v.push(&body.right_codes);
                        v.push(&body.right_scales);
                    }
                    MixedGpuWeight::Uniform(body) => {
                        v.push(&body.codes);
                        v.push(&body.scales);
                    }
                    MixedGpuWeight::Affine(body) => {
                        v.push(&body.codes);
                        v.push(&body.scales);
                        if let Some(b) = &body.biases {
                            v.push(b);
                        }
                    }
                }
            }
            v
        }

        pub fn q4_tensor_count(&self) -> usize {
            self.q4.len()
        }

        pub fn mixed_tensor_count(&self) -> usize {
            self.mixed.len()
        }

        pub fn f32_tensor_count(&self) -> usize {
            self.f32s.len()
        }
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(index, 4, &value as *const u32 as *const _);
    }

    fn tg256_grid(rows: u32) -> (u32, u32, u32) {
        (rows.saturating_mul(256), 1, 1)
    }

    fn simd8_grid(rows: u32) -> (u32, u32, u32) {
        (rows.div_ceil(8).saturating_mul(256).max(256), 1, 1)
    }

    pub struct Qwen38HybridWorkspace {
        hidden: PinnedBuffer,
        normalized: PinnedBuffer,
        qkvz: PinnedBuffer,
        ba: PinnedBuffer,
        repeated_q: PinnedBuffer,
        repeated_k: PinnedBuffer,
        conv_v: PinnedBuffer,
        z: PinnedBuffer,
        decay: PinnedBuffer,
        beta: PinnedBuffer,
        rec_out: PinnedBuffer,
        gated: PinnedBuffer,
        mixer: PinnedBuffer,
        first_residual: PinnedBuffer,
        q_proj: PinnedBuffer,
        k_proj: PinnedBuffer,
        v_proj: PinnedBuffer,
        query: PinnedBuffer,
        attn: PinnedBuffer,
        gated_attn: PinnedBuffer,
        gate: PinnedBuffer,
        up: PinnedBuffer,
        act: PinnedBuffer,
        down: PinnedBuffer,
        logits: PinnedBuffer,
        sampled: PinnedBuffer,
        argmax_part_v: PinnedBuffer,
        argmax_part_i: PinnedBuffer,
        conv_state: PinnedBuffer,
        rec_state: PinnedBuffer,
        gqa_key: PinnedBuffer,
        gqa_value: PinnedBuffer,
        hgravs_mid: PinnedBuffer,
        split_qkv: PinnedBuffer,
        split_b: PinnedBuffer,
        split_a: PinnedBuffer,
        xsum64: PinnedBuffer,
    }

    impl Qwen38HybridWorkspace {
        fn allocate(ctx: &MetalContext, max_seq_len: usize) -> Result<Self> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let f32b = |n: usize| {
                n.checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| Error::Model("qwen38 workspace overflow".into()))
            };
            let hidden = f32b(QWEN38_HIDDEN)?;
            let qkvz = f32b(layout.qkvz_rows())?;
            let ba = f32b(layout.ba_rows())?;
            let value = f32b(layout.value_elements())?;
            let q_proj = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2)?;
            let kv = f32b(QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
            let query = f32b(QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM)?;
            let mid = f32b(QWEN38_INTERMEDIATE)?;
            let logits = f32b(QWEN38_VOCAB)?;
            let conv = f32b(48 * layout.conv_state_elements())?;
            let rec = f32b(48 * layout.recurrent_state_elements())?;
            let kv_cache =
                f32b(QWEN38_GQA_LAYERS * max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM)?;
            Ok(Self {
                hidden: ctx.new_buffer_checked(hidden)?,
                normalized: ctx.new_buffer_checked(hidden)?,
                qkvz: ctx.new_buffer_checked(qkvz)?,
                ba: ctx.new_buffer_checked(ba)?,
                repeated_q: ctx.new_buffer_checked(value)?,
                repeated_k: ctx.new_buffer_checked(value)?,
                conv_v: ctx.new_buffer_checked(value)?,
                z: ctx.new_buffer_checked(value)?,
                decay: ctx.new_buffer_checked(f32b(layout.value_heads)?)?,
                beta: ctx.new_buffer_checked(f32b(layout.value_heads)?)?,
                rec_out: ctx.new_buffer_checked(value)?,
                gated: ctx.new_buffer_checked(value)?,
                mixer: ctx.new_buffer_checked(hidden)?,
                first_residual: ctx.new_buffer_checked(hidden)?,
                q_proj: ctx.new_buffer_checked(q_proj)?,
                k_proj: ctx.new_buffer_checked(kv)?,
                v_proj: ctx.new_buffer_checked(kv)?,
                query: ctx.new_buffer_checked(query)?,
                attn: ctx.new_buffer_checked(query)?,
                gated_attn: ctx.new_buffer_checked(query)?,
                gate: ctx.new_buffer_checked(mid)?,
                up: ctx.new_buffer_checked(mid)?,
                act: ctx.new_buffer_checked(mid)?,
                down: ctx.new_buffer_checked(hidden)?,
                logits: ctx.new_buffer_checked(logits)?,
                sampled: ctx.new_buffer_checked(std::mem::size_of::<u32>())?,
                // ARGMAX_GROUPS partials, one (value, index) pair per threadgroup
                argmax_part_v: ctx.new_buffer_checked(ARGMAX_GROUPS * 4)?,
                argmax_part_i: ctx.new_buffer_checked(ARGMAX_GROUPS * 4)?,
                conv_state: ctx.new_buffer_checked(conv)?,
                rec_state: ctx.new_buffer_checked(rec)?,
                gqa_key: ctx.new_buffer_checked(kv_cache)?,
                gqa_value: ctx.new_buffer_checked(kv_cache)?,
                hgravs_mid: ctx.new_buffer_checked(f32b(QWEN38_MIXED_HGRAVS_RANK)?)?,
                split_qkv: ctx.new_buffer_checked(f32b(
                    crate::model::qwen38_geometry::QWEN38_IN_PROJ_QKV_ROWS,
                )?)?,
                split_b: ctx.new_buffer_checked(f32b(
                    crate::model::qwen38_geometry::QWEN38_IN_PROJ_B_ROWS,
                )?)?,
                split_a: ctx.new_buffer_checked(f32b(
                    crate::model::qwen38_geometry::QWEN38_IN_PROJ_A_ROWS,
                )?)?,
                xsum64: ctx.new_buffer_checked(f32b(QWEN38_XSUM64_CAP)?)?,
            })
        }

        fn resident_bytes(&self) -> u64 {
            [
                &self.hidden,
                &self.normalized,
                &self.qkvz,
                &self.ba,
                &self.repeated_q,
                &self.repeated_k,
                &self.conv_v,
                &self.z,
                &self.decay,
                &self.beta,
                &self.rec_out,
                &self.gated,
                &self.mixer,
                &self.first_residual,
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                &self.query,
                &self.attn,
                &self.gated_attn,
                &self.gate,
                &self.up,
                &self.act,
                &self.down,
                &self.logits,
                &self.sampled,
                &self.conv_state,
                &self.rec_state,
                &self.gqa_key,
                &self.gqa_value,
                &self.hgravs_mid,
                &self.split_qkv,
                &self.split_b,
                &self.split_a,
                &self.xsum64,
            ]
            .iter()
            .map(|b| b.length())
            .sum()
        }
    }

    /// Names used by the per-token graph.  The catalog maps by owned tensor
    /// names, but those names are immutable after load.  Keeping one owned
    /// copy per layer lets the decode loop pass borrowed `&str`s to the
    /// pipeline cache instead of formatting and dropping hundreds of
    /// `String`s on every token.
    struct Qwen38LayerNameCache {
        names: Vec<String>,
    }

    impl Qwen38LayerNameCache {
        // Keep invalid diagnostic/probe inputs on the normal missing-weight
        // error path.  The hot graph only supplies in-range layers, but the
        // public parity helpers accept a caller-provided index and the old
        // formatter did not turn that into an out-of-bounds access.
        const INVALID_NAME: &'static str = "language_model.model.layers.__invalid__";

        const SUFFIXES: [&'static str; 23] = [
            "input_layernorm.weight",
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_ba.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_qkvz.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            "mlp.down_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "post_attention_layernorm.weight",
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_norm.weight",
            "self_attn.v_proj.weight",
        ];

        fn new() -> Self {
            let mut names = Vec::with_capacity(QWEN38_LAYERS * Self::SUFFIXES.len());
            for layer in 0..QWEN38_LAYERS {
                for suffix in Self::SUFFIXES {
                    names.push(format!("language_model.model.layers.{layer}.{suffix}"));
                }
            }
            Self { names }
        }

        #[inline]
        fn get(&self, layer: usize, suffix: &str) -> &str {
            if layer >= QWEN38_LAYERS {
                return Self::INVALID_NAME;
            }
            let slot = match suffix {
                "input_layernorm.weight" => 0,
                "linear_attn.A_log" => 1,
                "linear_attn.conv1d.weight" => 2,
                "linear_attn.dt_bias" => 3,
                "linear_attn.in_proj_a.weight" => 4,
                "linear_attn.in_proj_b.weight" => 5,
                "linear_attn.in_proj_ba.weight" => 6,
                "linear_attn.in_proj_qkv.weight" => 7,
                "linear_attn.in_proj_qkvz.weight" => 8,
                "linear_attn.in_proj_z.weight" => 9,
                "linear_attn.norm.weight" => 10,
                "linear_attn.out_proj.weight" => 11,
                "mlp.down_proj.weight" => 12,
                "mlp.gate_proj.weight" => 13,
                "mlp.up_proj.weight" => 14,
                "post_attention_layernorm.weight" => 15,
                "self_attn.k_norm.weight" => 16,
                "self_attn.k_proj.weight" => 17,
                "self_attn.o_proj.weight" => 18,
                "self_attn.q_norm.weight" => 19,
                "self_attn.q_proj.weight" => 20,
                "self_attn.v_norm.weight" => 21,
                "self_attn.v_proj.weight" => 22,
                _ => panic!("unknown qwen38 layer-name suffix {suffix}"),
            };
            &self.names[layer * Self::SUFFIXES.len() + slot]
        }
    }

    #[cfg(test)]
    mod layer_name_cache_tests {
        use super::*;

        #[test]
        fn cached_names_match_catalog_authority() {
            let cache = Qwen38LayerNameCache::new();
            for layer in 0..QWEN38_LAYERS {
                for suffix in Qwen38LayerNameCache::SUFFIXES {
                    assert_eq!(
                        cache.get(layer, suffix),
                        qwen38_layer_name(layer, suffix),
                        "cached qwen38 name drifted for layer {layer} suffix {suffix}"
                    );
                }
            }
            assert_eq!(
                cache.get(QWEN38_LAYERS, "mlp.gate_proj.weight"),
                Qwen38LayerNameCache::INVALID_NAME
            );
        }
    }

    pub struct Qwen38HybridDecodeSession {
        context: MetalContext,
        weights: Arc<Qwen38HybridWeights>,
        workspace: Qwen38HybridWorkspace,
        max_seq_len: usize,
        position: usize,
        /// `dispatch_threads` labels harvested when
        /// `HAWKING_TRACE_DISPATCH=1`, WITH THEIR COUNTS. `None` on the
        /// default path so `step` allocates nothing extra.
        ///
        /// A set here would have been enough for the fusion sentinels that
        /// first needed this, and it is NOT enough to answer how many
        /// dispatches a token costs -- the runtime pushes one label per
        /// dispatch and collapsing them to a set destroys exactly the
        /// multiplicity the question is about.
        seen_kernels: Option<BTreeMap<String, u64>>,
        /// Snapshot of the default-on packed reconstruction path at attach.
        /// Keeping this beside the other session controls avoids reparsing the
        /// environment for every packed GEMV.
        recon_fuse: bool,
        /// Snapshot of the shared family-name selector at session attach.
        /// Legacy `=0` names remain available for controlled A/B runs without
        /// reparsing the environment for every mixed dispatch.
        decode_family: bool,
        pub fallbacks: u32,
        /// Default matches the shipped bring-up binding. Diagnostic lanes may
        /// retarget to another shipped kernel; they must not invent one.
        pub matvec_kernel: Qwen38MatvecKernel,
        /// Overlap independent projections (gate+up, qkvz+ba, q/k/v) in one
        /// concurrent encoder. Off by default so `step` stays bit-identical
        /// to the bring-up vehicle.
        pub concurrent_independent: bool,
        /// Launch one threadgroup per (value-head, value-dim) for the
        /// gated-delta recurrence. Same serial reduction as the Q80 kernel;
        /// the vi columns are independent. Default ON after paired generate
        /// admitted a 42.7→33.4 ms token cut with greedy-identical ids.
        pub deltanet_vi_parallel: bool,
        /// MLP suffix fusion. Default Off. See [`Qwen38MlpFusion`].
        pub mlp_fusion: Qwen38MlpFusion,
        /// Fuse GQA Q/K/V into one geo_tpr64 concat dispatch. Default Off.
        pub fuse_gqa_qkv: bool,
        /// Fuse Qwen3.8 MHA output with the per-head sigmoid gate. Default Off.
        pub fuse_attention_gate: bool,
        /// Fuse DeltaNet qkvz+ba into one geo_tpr64 concat dispatch. Default Off.
        pub fuse_dn_inproj: bool,
        /// Fuse residual add + the following RMSNorm. Default Off.
        pub fuse_add_rmsnorm: bool,
        /// BAD control: fused kernel multiplies by weight[i] not (1+w).
        pub fuse_add_rmsnorm_bad: bool,
        /// Fuse ba_to_decay into gated-delta. Default Off.
        pub fuse_ba_delta: bool,
        /// BAD control: fused kernel uses identity decay/beta.
        pub fuse_ba_delta_bad: bool,
        /// Gated-delta state kernel. Default Baseline (N025 vi_simd_ba).
        pub dn_state_kernel: Qwen38DeltaNetStateKernel,
        /// Affine2 GEMV geometry. Default tpr64 (incumbent / no-op control).
        pub affine2_geo: Affine2Geo,
        /// Q2F GEMV geometry. Unset environment inherits `affine2_geo` for
        /// historical behavior; an explicit Q2F value isolates the biasless
        /// representation in protected A/B runs.
        pub q2f_geo: Affine2Geo,
        /// GEMV weights reconstructed to dense float. Copied from the catalog
        /// load census and incremented only by [`Self::account_dense_w`].
        pub dense_w_materialized: u64,
        /// BAD control: force the serial one-thread-per-row q2f kernel.
        pub q2f_force_serial: bool,
        /// One serial compute encoder for the whole token graph. Default off
        /// (one encoder per dispatch). Opt-in attack on encoder-boundary idle
        /// and host command construction. Independent of `concurrent_independent`.
        pub serial_token_encoder: bool,
        /// Validated launch geometry captured at session attach. Keeping
        /// these outside the layer loop removes repeated environment parsing
        /// from the token-encoding hot path.
        rmsnorm_tg: u32,
        dn_rmsnorm_tg: u32,
        rope_tg: u32,
        /// Process-scoped DeltaNet and sampling controls, resolved once when
        /// the resident session attaches.
        deltanet_vi_simd: bool,
        argmax_two_pass: bool,
        /// Sum of packed weight payloads bound by the current token graph.
        /// This is an active-payload denominator, not a hardware read counter.
        active_weight_bytes: Cell<u64>,
        /// Keep active-byte accounting out of the explicitly untimed serving
        /// route. Measured and protected callers leave this enabled so the
        /// traffic contract remains unchanged.
        active_weight_accounting: bool,
        /// Immutable tensor names reused by every token graph.
        layer_names: Qwen38LayerNameCache,
        /// Validated mixer kind for each layer. The source schedule is fixed
        /// for Qwen3.8, so decoding it once at attach removes repeated modulo
        /// and error-path construction from the token graph.
        mixer_kinds: [Qwen38MixerKind; QWEN38_LAYERS],
    }

    impl Qwen38HybridDecodeSession {
        pub fn open(root: impl AsRef<Path>, max_seq_len: usize) -> Result<Self> {
            Self::attach(Arc::new(Qwen38HybridWeights::load(root)?), max_seq_len)
        }

        /// New decode session against an already-resident weight set.
        /// Allocates only workspace / KV / DeltaNet state.
        pub fn attach(weights: Arc<Qwen38HybridWeights>, max_seq_len: usize) -> Result<Self> {
            qwen38_assert_schedule_intact()?;
            if max_seq_len == 0 {
                return Err(Error::Model("qwen38 max_seq_len must be positive".into()));
            }
            let workspace = Qwen38HybridWorkspace::allocate(&weights.context, max_seq_len)?;
            let expected = qwen38_workspace_bytes(max_seq_len)?;
            let got = workspace.resident_bytes();
            if got != expected.total_bytes as u64 {
                return Err(Error::Model(format!(
                    "qwen38 workspace bytes {got} != formula {}",
                    expected.total_bytes
                )));
            }
            zero_buffer(&workspace.conv_state);
            zero_buffer(&workspace.rec_state);
            zero_buffer(&workspace.gqa_key);
            zero_buffer(&workspace.gqa_value);
            let dense_w_materialized = weights.dense_w_materialized;
            let mixer_kinds = std::array::from_fn(|layer| {
                if (layer + 1) % QWEN38_FULL_ATTENTION_INTERVAL == 0 {
                    Qwen38MixerKind::Gqa
                } else {
                    Qwen38MixerKind::DeltaNet
                }
            });
            let recon_fuse = qwen38_recon_fuse_enabled();
            let decode_family = crate::decode_family::family_dispatch_enabled();
            let concurrent_independent = qwen38_concurrent_independent_enabled();
            let (fuse_add_rmsnorm, fuse_add_rmsnorm_bad) =
                qwen38_fuse_add_rmsnorm_from_env();
            let (fuse_ba_delta, fuse_ba_delta_bad) = qwen38_fuse_ba_delta_from_env();
            let session = Self {
                context: weights.context.clone(),
                weights,
                workspace,
                max_seq_len,
                position: 0,
                seen_kernels: if qwen38_trace_dispatch_enabled() {
                    Some(BTreeMap::new())
                } else {
                    None
                },
                recon_fuse,
                decode_family,
                fallbacks: 0,
                matvec_kernel: Qwen38MatvecKernel::from_env(),
                concurrent_independent,
                deltanet_vi_parallel: true,
                mlp_fusion: Qwen38MlpFusion::from_env(),
                fuse_gqa_qkv: qwen38_fuse_gqa_qkv_enabled(),
                fuse_attention_gate: qwen38_fuse_attention_gate_enabled(),
                fuse_dn_inproj: qwen38_fuse_dn_inproj_enabled(),
                fuse_add_rmsnorm,
                fuse_add_rmsnorm_bad,
                fuse_ba_delta,
                fuse_ba_delta_bad,
                dn_state_kernel: Qwen38DeltaNetStateKernel::from_env(),
                affine2_geo: Affine2Geo::from_env(),
                q2f_geo: qwen38_q2f_geo_from_env(),
                dense_w_materialized,
                q2f_force_serial: false,
                // A serial encoder is the fast profile's low-risk host-side
                // candidate. An explicitly requested concurrent wave wins so
                // the two encoder topologies are never nested.
                serial_token_encoder: qwen38_serial_token_encoder_enabled()
                    && !concurrent_independent,
                rmsnorm_tg: qwen38_rmsnorm_tg_from_env(),
                dn_rmsnorm_tg: qwen38_dn_rmsnorm_tg_from_env(),
                rope_tg: qwen38_rope_tg_from_env(),
                deltanet_vi_simd: crate::env_opt_out("HAWKING_DN_VI_SIMD"),
                argmax_two_pass: std::env::var("HAWKING_ARGMAX_TWO_PASS")
                    .map(|v| v != "0")
                    .unwrap_or(false),
                active_weight_bytes: Cell::new(0),
                active_weight_accounting: true,
                layer_names: Qwen38LayerNameCache::new(),
                mixer_kinds,
            };
            // `MetalContext` clones share `Arc<DispatchTrace>`. If that ever
            // becomes a fresh buffer, `drain_trace` on the session would miss
            // every TCB sample recorded against the weight-load context.
            assert!(
                std::sync::Arc::ptr_eq(&session.context.trace, &session.weights.context.trace),
                "qwen38 dispatch trace must survive MetalContext clone"
            );
            let allocs = session.weights.residency_allocations();
            session.context.request_residency(&allocs)?;
            Ok(session)
        }

        pub fn share_weights(&self) -> Arc<Qwen38HybridWeights> {
            Arc::clone(&self.weights)
        }

        pub fn weights(&self) -> &Qwen38HybridWeights {
            &self.weights
        }

        pub fn max_seq_len(&self) -> usize {
            self.max_seq_len
        }

        pub fn workspace_resident_bytes(&self) -> u64 {
            self.workspace.resident_bytes()
        }

        pub fn resident_weight_bytes(&self) -> u64 {
            self.weights.resident_bytes()
        }

        pub fn last_active_weight_bytes(&self) -> u64 {
            self.active_weight_bytes.get()
        }

        #[inline]
        fn layer_name(&self, layer: usize, suffix: &str) -> &str {
            self.layer_names.get(layer, suffix)
        }

        #[inline]
        fn mixer_kind(&self, layer: usize) -> Result<Qwen38MixerKind> {
            self.mixer_kinds.get(layer).copied().ok_or_else(|| {
                Error::Model(format!(
                    "qwen38 layer {layer} is outside 0..{QWEN38_LAYERS}"
                ))
            })
        }

        #[inline]
        fn deltanet_state_slot(&self, layer: usize) -> Result<usize> {
            match self.mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => {
                    Ok(layer - layer / QWEN38_FULL_ATTENTION_INTERVAL)
                }
                Qwen38MixerKind::Gqa => Err(Error::Model(format!(
                    "qwen38 layer {layer} is GQA; DeltaNet slot is undefined"
                ))),
            }
        }

        #[inline]
        fn gqa_state_slot(&self, layer: usize) -> Result<usize> {
            match self.mixer_kind(layer)? {
                Qwen38MixerKind::Gqa => Ok(layer / QWEN38_FULL_ATTENTION_INTERVAL),
                Qwen38MixerKind::DeltaNet => Err(Error::Model(format!(
                    "qwen38 layer {layer} is DeltaNet; GQA slot is undefined"
                ))),
            }
        }

        pub fn shares_weights_with(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.weights, &other.weights)
        }

        fn enable_dispatch_name_trace(&self, tcb: &mut TokenCommandBuffer<'_>) {
            if self.seen_kernels.is_some() {
                // Requires TCB Off. Cpu/gpu timing modes refuse this opt-in
                // and instead flush remapped names into `ctx.trace`.
                let _ = tcb.enable_structural_kernel_trace();
            }
        }

        fn harvest_dispatch_names(&mut self, names: Option<Vec<String>>) {
            let Some(seen) = self.seen_kernels.as_mut() else {
                return;
            };
            if let Some(names) = names {
                for name in names {
                    *seen.entry(name).or_insert(0) += 1;
                }
            }
        }

        /// Distinct kernel names the runtime actually dispatched since the
        /// last drain (or session open). Empty when
        /// `HAWKING_TRACE_DISPATCH` is not `1`.
        ///
        /// Primary source is the TCB structural label list (the exact
        /// `dispatch_threads` string, no `static_kernel_name` remap).
        /// Also unions `MetalContext::drain_trace` so a
        /// `HAWKING_TCB_TRACE=cpu` run still reports through the
        /// `Arc<DispatchTrace>` that survives `weights.context.clone()`.
        /// Per-kernel dispatch COUNTS since the last drain, from the TCB
        /// structural label list -- one entry pushed per `dispatch_threads`.
        ///
        /// Does NOT drain, and does NOT union `MetalContext::drain_trace`:
        /// that path is a timing sampler and unioning it would double-count.
        /// Read this BEFORE `drain_dispatched_kernel_names`, which clears the
        /// same store. Empty when `HAWKING_TRACE_DISPATCH` is not `1`.
        pub fn dispatched_kernel_histogram(&self) -> Vec<(String, u64)> {
            let Some(seen) = self.seen_kernels.as_ref() else {
                return Vec::new();
            };
            let mut rows: Vec<(String, u64)> =
                seen.iter().map(|(k, v)| (k.clone(), *v)).collect();
            rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            rows
        }

        pub fn drain_dispatched_kernel_names(&mut self) -> Vec<String> {
            let mut names: Vec<String> = self
                .context
                .drain_trace()
                .into_iter()
                .map(|sample| sample.kernel_name.to_string())
                .filter(|name| name != "other" && !name.starts_with("tcb_"))
                .collect();
            if let Some(seen) = self.seen_kernels.as_mut() {
                names.extend(std::mem::take(seen).into_keys());
            }
            names.sort();
            names.dedup();
            names
        }



        fn assert_mixed_mlp_native(mixed: &HashMap<String, MixedGpuWeight>) -> Result<()> {
            assert_mixed_mlp_native_kinds(|name| {
                mixed.get(name).map(|weight| match weight {
                    MixedGpuWeight::Binary(_) => MixedMlpNativeKind::Binary,
                    MixedGpuWeight::Residual(_) => MixedMlpNativeKind::Residual,
                    MixedGpuWeight::Hgravs(_) => MixedMlpNativeKind::Hgravs,
                    MixedGpuWeight::Uniform(_) => MixedMlpNativeKind::Uniform,
                    MixedGpuWeight::Affine(_) => MixedMlpNativeKind::AffineScaleBias,
                })
            })
        }

        fn upload_mixed(
            context: &MetalContext,
            codec: u8,
            payload: &[u8],
            name: &str,
        ) -> Result<MixedGpuWeight> {
            let expected = match codec {
                0 => &MAGIC_BINARY[..],
                1 => &MAGIC_RESIDUAL_COMPACT[..],
                2 => &MAGIC_HGRAVS01[..],
                3 => &MAGIC_UNIFORM[..],
                other => {
                    return Err(mixed_error(format!(
                        "{name} upload unknown codec {other}"
                    )))
                }
            };
            if payload.len() < 8 || payload[..8] != *expected {
                return Err(mixed_error(format!(
                    "{name} codec {codec} magic {:?} != {:?}; refusing silent fallback",
                    payload.get(..8),
                    expected
                )));
            }
            let layout = mixed_gpu_layout(codec, payload)?;
            match layout.kind {
                MixedGpuKind::Binary {
                    scale_off,
                    scale_bytes,
                    sign_off,
                    sign_bytes,
                    group_size,
                    groups_per_row,
                } => Ok(MixedGpuWeight::Binary(GpuBinary {
                    signs: context
                        .new_buffer_with_bytes_checked(&payload[sign_off..sign_off + sign_bytes])?,
                    scales: context
                        .new_buffer_with_bytes_checked(&payload[scale_off..scale_off + scale_bytes])?,
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
                        return Err(mixed_error(format!("{name} residual binary layout drifted")));
                    };
                    let rice = RiceQ1Packed {
                        binary: BinaryGroupPacked {
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
                    let row_ptr =
                        rice_q1_row_ptr(&indices, layout.rows as usize, layout.cols as usize)?;
                    let idx_bytes: Vec<u8> =
                        indices.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let ptr_bytes: Vec<u8> =
                        row_ptr.iter().flat_map(|v| v.to_le_bytes()).collect();
                    Ok(MixedGpuWeight::Residual(GpuResidual {
                        binary: GpuBinary {
                            signs: context.new_buffer_with_bytes_checked(
                                &payload[*sign_off..*sign_off + *sign_bytes],
                            )?,
                            scales: context.new_buffer_with_bytes_checked(
                                &payload[*scale_off..*scale_off + *scale_bytes],
                            )?,
                            rows: layout.rows,
                            cols: layout.cols,
                            group_size: *group_size,
                            groups_per_row: *groups_per_row,
                        },
                        indices: context.new_buffer_with_bytes_checked(&idx_bytes)?,
                        row_ptr: context.new_buffer_with_bytes_checked(&ptr_bytes)?,
                        residual_signs: context.new_buffer_with_bytes_checked(
                            &payload[residual_sign_off..residual_sign_off + residual_sign_bytes],
                        )?,
                        residual_scale_f16,
                    }))
                }
                MixedGpuKind::Hgravs { left, right } => {
                    if left.bits != u32::from(QWEN38_MIXED_HGRAVS_BITS)
                        || right.bits != u32::from(QWEN38_MIXED_HGRAVS_BITS)
                        || left.group_size != QWEN38_MIXED_HGRAVS_GROUP as u32
                        || right.group_size != QWEN38_MIXED_HGRAVS_GROUP as u32
                        || left.cols != QWEN38_MIXED_HGRAVS_RANK as u32
                        || right.rows != QWEN38_MIXED_HGRAVS_RANK as u32
                    {
                        return Err(mixed_error(format!(
                            "{name} HGRAVS01 geometry {}x{} / {}x{} bits={}/{} group={}/{} is not r160_b3",
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
                    Ok(MixedGpuWeight::Hgravs(GpuHgravs {
                        left_codes: context.new_buffer_with_bytes_checked(
                            &payload[left.code_off..left.code_off + left.code_bytes],
                        )?,
                        left_scales: context.new_buffer_with_bytes_checked(
                            &payload[left.scale_off..left.scale_off + left.scale_bytes],
                        )?,
                        right_codes: context.new_buffer_with_bytes_checked(
                            &payload[right.code_off..right.code_off + right.code_bytes],
                        )?,
                        right_scales: context.new_buffer_with_bytes_checked(
                            &payload[right.scale_off..right.scale_off + right.scale_bytes],
                        )?,
                        left_rows: left.rows,
                        left_cols: left.cols,
                        right_rows: right.rows,
                        right_cols: right.cols,
                        group_size: left.group_size,
                        bits: left.bits,
                        bound: left.bound,
                    }))
                }
                MixedGpuKind::Uniform(factor) => Ok(MixedGpuWeight::Uniform(GpuUniform {
                    codes: context.new_buffer_with_bytes_checked(
                        &payload[factor.code_off..factor.code_off + factor.code_bytes],
                    )?,
                    scales: context.new_buffer_with_bytes_checked(
                        &payload[factor.scale_off..factor.scale_off + factor.scale_bytes],
                    )?,
                    rows: factor.rows,
                    cols: factor.cols,
                    group_size: factor.group_size,
                    bits: factor.bits,
                    bound: factor.bound,
                })),
                MixedGpuKind::Affine { .. } => Err(mixed_error(format!(
                    "{name} HGRAVF01 is a MixedCatalogLane, not an upload_mixed codec"
                ))),
            }
        }

        pub fn reset(&mut self) {
            self.position = 0;
            self.reset_active_weight_bytes();
            zero_buffer(&self.workspace.conv_state);
            zero_buffer(&self.workspace.rec_state);
            zero_buffer(&self.workspace.gqa_key);
            zero_buffer(&self.workspace.gqa_value);
        }

        fn q4(&self, name: &str) -> Result<&Q4Weight> {
            self.weights
                .q4
                .get(name)
                .ok_or_else(|| Error::Model(format!("qwen38 missing Q4 {name}")))
        }

        fn affine(&self, name: &str) -> Option<&GpuAffine> {
            match self.weights.mixed.get(name) {
                Some(MixedGpuWeight::Affine(body)) => Some(body),
                _ => None,
            }
        }

        /// A fused path whose kernel does not exist must not be selected.
        ///
        /// The static preflight reports both fused Q2F kernels as
        /// kernel_existence ERRORs: the host names
        /// qwen_q2f_group64_matvec_qkv_geo_tpr64_tg128 and
        /// ..._pair_geo_tpr64_tg128 and no shader defines either. The sibling
        /// family in q80_mixed_decode.metal stops one short of both.
        ///
        /// sealed-3.14 never takes these branches - its GQA q/k/v are codec 3
        /// (uniform q4), so the q4 fusion wins first - which is why this has
        /// been latent rather than a crash. But the gate is `bits == 2`, and
        /// this campaign is actively pursuing 2-bit bodies. The first artifact
        /// with 2-bit attention would select a pipeline that cannot be built.
        ///
        /// So the fused path requires its kernel to EXIST. Absent it, the caller
        /// falls through to the per-tensor matvec, which is correct and merely
        /// unfused - a real fallback rather than an impossible dispatch, and no
        /// impossible runtime path enters Odyssey.
        fn q2f_kernel_available(&self, name: &str) -> bool {
            self.context.pipeline(name).is_ok()
        }

        fn can_fuse_q2f_qkv(&self, layer: usize) -> bool {
            if !self.q2f_kernel_available(QWEN38_Q2F_QKV_GEO_KERNEL) {
                return false;
            }
            let q_name = self.layer_name(layer, "self_attn.q_proj.weight");
            let k_name = self.layer_name(layer, "self_attn.k_proj.weight");
            let v_name = self.layer_name(layer, "self_attn.v_proj.weight");
            let (Some(q), Some(k), Some(v)) = (
                self.affine(&q_name),
                self.affine(&k_name),
                self.affine(&v_name),
            ) else {
                return false;
            };
            q.rows > 0
                && k.rows > 0
                && v.rows > 0
                && q.cols == k.cols
                && q.cols == v.cols
                && q.cols % 64 == 0
                && [q, k, v].iter().all(|body| {
                    body.bits == 2
                        && body.group_size == 64
                        && body.biases.is_none()
                })
        }

        /// Same rule as can_fuse_q2f_qkv: no kernel, no fused path. See there.
        fn can_fuse_q2f_pair(&self, layer: usize) -> bool {
            if !self.q2f_kernel_available(QWEN38_Q2F_PAIR_GEO_KERNEL) {
                return false;
            }
            let qkvz_name = self.layer_name(layer, "linear_attn.in_proj_qkvz.weight");
            let ba_name = self.layer_name(layer, "linear_attn.in_proj_ba.weight");
            let (Some(qkvz), Some(ba)) = (self.affine(&qkvz_name), self.affine(&ba_name)) else {
                return false;
            };
            qkvz.rows > 0
                && ba.rows > 0
                && qkvz.cols == ba.cols
                && qkvz.cols % 64 == 0
                && [qkvz, ba].iter().all(|body| {
                    body.bits == 2 && body.group_size == 64 && body.biases.is_none()
                })
        }

        fn f32(&self, name: &str) -> Result<&PinnedBuffer> {
            let weight = self
                .weights
                .f32s
                .get(name)
                .ok_or_else(|| Error::Model(format!("qwen38 missing f32 {name}")))?;
            self.record_f32_weight(weight);
            Ok(weight)
        }

        fn add_active_weight_bytes(&self, bytes: u64) {
            if !self.active_weight_accounting {
                return;
            }
            self.active_weight_bytes
                .set(self.active_weight_bytes.get().saturating_add(bytes));
        }

        fn bytes_per_row(rows: u64, lengths: &[u64]) -> u64 {
            if rows == 0 {
                return 0;
            }
            lengths.iter().copied().fold(0u64, |total, length| {
                total.saturating_add(length.saturating_add(rows - 1) / rows)
            })
        }

        fn record_q4_weight(&self, weight: &Q4Weight) {
            self.add_active_weight_bytes(
                weight
                    .codes
                    .length()
                    .saturating_add(weight.scales.length()),
            );
        }

        fn record_q4_embedding_row(&self, weight: &Q4Weight) {
            self.add_active_weight_bytes(Self::bytes_per_row(
                weight.rows as u64,
                &[weight.codes.length(), weight.scales.length()],
            ));
        }

        fn record_f32_weight(&self, weight: &PinnedBuffer) {
            self.add_active_weight_bytes(weight.length());
        }

        fn record_mixed_weight(&self, weight: &MixedGpuWeight) {
            self.add_active_weight_bytes(weight.resident_bytes());
        }

        fn record_affine_weight(&self, weight: &GpuAffine) {
            self.add_active_weight_bytes(
                weight
                    .codes
                    .length()
                    .saturating_add(weight.scales.length())
                    .saturating_add(weight.biases.as_ref().map(|buffer| buffer.length()).unwrap_or(0)),
            );
        }

        fn record_uniform_embedding_row(&self, weight: &GpuUniform) {
            self.add_active_weight_bytes(Self::bytes_per_row(
                weight.rows as u64,
                &[weight.codes.length(), weight.scales.length()],
            ));
        }

        fn record_affine_embedding_row(&self, weight: &GpuAffine) {
            let bias_bytes = weight.biases.as_ref().map(|buffer| buffer.length()).unwrap_or(0);
            self.add_active_weight_bytes(Self::bytes_per_row(
                weight.rows as u64,
                &[weight.codes.length(), weight.scales.length(), bias_bytes],
            ));
        }

        fn reset_active_weight_bytes(&self) {
            self.active_weight_bytes.set(0);
        }

        fn encode_q4_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            // The geo enum's as_str is a const fn, so the bitcast swap happens
            // here at the dispatch rather than inside it.
            self.encode_q4_matvec_kernel(
                tcb,
                name,
                input,
                output,
                qwen38_q4_kernel(self.matvec_kernel.as_str()),
            )
        }

        fn encode_named_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            // HQ38M20 keeps native packed weights in `mixed` and only uses
            // `q4` for explicitly declared HQ30UQ4 fallback islands. Probe
            // the native map first in that shape; the non-mixed catalog keeps
            // the historical Q4-first order. This removes one guaranteed
            // failed hash probe from every common mixed-catalog GEMV without
            // changing behavior for either catalog layout.
            if !self.weights.mixed.is_empty() {
                if let Some(weight) = self.weights.mixed.get(name) {
                    return self.encode_mixed_weight(tcb, weight, input, output);
                }
                if let Some(weight) = self.weights.q4.get(name) {
                    return self.encode_q4_matvec_weight(
                        tcb,
                        name,
                        weight,
                        input,
                        output,
                        qwen38_q4_kernel(self.matvec_kernel.as_str()),
                    );
                }
            } else if let Some(weight) = self.weights.q4.get(name) {
                return self.encode_q4_matvec_weight(
                    tcb,
                    name,
                    weight,
                    input,
                    output,
                    qwen38_q4_kernel(self.matvec_kernel.as_str()),
                );
            }
            Err(mixed_error(format!(
                "missing GEMV {name}; refusing silent reconstructed or dense path"
            )))
        }

        fn encode_mixed_weight(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            weight: &MixedGpuWeight,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            self.record_mixed_weight(weight);
            match weight {
                MixedGpuWeight::Binary(body) => self.dispatch_binary(tcb, body, input, output),
                MixedGpuWeight::Residual(body) => self.dispatch_residual(tcb, body, input, output),
                MixedGpuWeight::Hgravs(body) => self.dispatch_hgravs(tcb, body, input, output),
                MixedGpuWeight::Uniform(body) => self.dispatch_uniform(tcb, body, input, output),
                MixedGpuWeight::Affine(body) => self.dispatch_affine(tcb, body, input, output),
            }
        }

        fn encode_binary_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            packed: &GpuBinary,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            enc.set_buffer(0, Some(&packed.signs), 0);
            enc.set_buffer(1, Some(&packed.scales), 0);
            enc.set_buffer(2, Some(input), 0);
            enc.set_buffer(3, Some(output), 0);
            set_u32(enc, 4, packed.rows);
            set_u32(enc, 5, packed.cols);
            set_u32(enc, 6, packed.group_size);
            set_u32(enc, 7, packed.groups_per_row);
        }

        fn encode_binary_csr_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            packed: &GpuResidual,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            enc.set_buffer(0, Some(&packed.binary.signs), 0);
            enc.set_buffer(1, Some(&packed.binary.scales), 0);
            enc.set_buffer(2, Some(input), 0);
            enc.set_buffer(3, Some(output), 0);
            enc.set_buffer(4, Some(&packed.indices), 0);
            enc.set_buffer(5, Some(&packed.row_ptr), 0);
            enc.set_buffer(6, Some(&packed.residual_signs), 0);
            set_u32(enc, 7, packed.binary.rows);
            set_u32(enc, 8, packed.binary.cols);
            set_u32(enc, 9, packed.binary.group_size);
            set_u32(enc, 10, packed.binary.groups_per_row);
            set_u32(enc, 11, packed.residual_scale_f16);
        }

        fn encode_csr_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            packed: &GpuResidual,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            enc.set_buffer(0, Some(&packed.indices), 0);
            enc.set_buffer(1, Some(&packed.row_ptr), 0);
            enc.set_buffer(2, Some(&packed.residual_signs), 0);
            enc.set_buffer(3, Some(input), 0);
            enc.set_buffer(4, Some(output), 0);
            set_u32(enc, 5, packed.binary.rows);
            set_u32(enc, 6, packed.binary.cols);
            set_u32(enc, 7, packed.residual_scale_f16);
        }

        fn encode_factor_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            codes: &PinnedBuffer,
            scales: &PinnedBuffer,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            rows: u32,
            cols: u32,
            group_size: u32,
            bits: u32,
            bound: u32,
        ) {
            enc.set_buffer(0, Some(codes), 0);
            enc.set_buffer(1, Some(scales), 0);
            enc.set_buffer(2, Some(input), 0);
            enc.set_buffer(3, Some(output), 0);
            set_u32(enc, 4, rows);
            set_u32(enc, 5, cols);
            set_u32(enc, 6, group_size);
            set_u32(enc, 7, bits);
            set_u32(enc, 8, bound);
        }

        fn encode_affine_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            body: &GpuAffine,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            let biases = body
                .biases
                .as_ref()
                .expect("affine2 encode requires a bias buffer");
            enc.set_buffer(0, Some(&body.codes), 0);
            enc.set_buffer(1, Some(&body.scales), 0);
            enc.set_buffer(2, Some(biases), 0);
            enc.set_buffer(3, Some(input), 0);
            enc.set_buffer(4, Some(output), 0);
            set_u32(enc, 5, body.rows);
            set_u32(enc, 6, body.cols);
            set_u32(enc, 7, body.group_size);
        }

        fn encode_q2f_args(
            &self,
            enc: &metal::ComputeCommandEncoderRef,
            body: &GpuAffine,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) {
            enc.set_buffer(0, Some(&body.codes), 0);
            enc.set_buffer(1, Some(&body.scales), 0);
            enc.set_buffer(2, Some(input), 0);
            enc.set_buffer(3, Some(output), 0);
            set_u32(enc, 4, body.rows);
            set_u32(enc, 5, body.cols);
        }

        fn dispatch_q2f(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuAffine,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if body.bits != 2 || body.group_size != 64 || body.cols % 64 != 0 {
                return Err(mixed_error(format!(
                    "q2f dispatch refuses bits={} group_size={} cols={}",
                    body.bits, body.group_size, body.cols
                )));
            }
            if !self.q2f_force_serial && self.recon_fuse {
                let (name, grid, tg) = qwen38_q2f_matvec_launch(self.q2f_geo, body.rows);
                return tcb.dispatch_threads(name, grid, tg, |enc| {
                    self.encode_q2f_args(enc, body, input, output)
                });
            }
            tcb.dispatch_threads(
                QWEN38_Q2F_SERIAL,
                (body.rows, 1, 1),
                (256, 1, 1),
                |enc| self.encode_q2f_args(enc, body, input, output),
            )
        }

        fn dispatch_affine(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuAffine,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if body.biases.is_none() {
                return self.dispatch_q2f(tcb, body, input, output);
            }
            if body.bits != 2 || !affine_group_size_supported(body.group_size as usize) {
                return Err(mixed_error(format!(
                    "HGRAVF01 dispatch refuses bits={} group_size={}",
                    body.bits, body.group_size
                )));
            }
            if let Some((name, grid, tg)) = qwen38_affine_q2_launch_with_recon_fuse(
                self.affine2_geo,
                body.group_size,
                body.rows,
                body.cols,
                self.recon_fuse,
            ) {
                return tcb.dispatch_threads(name, grid, tg, |enc| {
                    self.encode_affine_args(enc, body, input, output)
                });
            }
            tcb.dispatch_threads(
                QWEN38_AFFINE_Q2_SERIAL,
                (body.rows, 1, 1),
                (256, 1, 1),
                |enc| self.encode_affine_args(enc, body, input, output),
            )
        }

        fn dispatch_binary(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuBinary,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if self.recon_fuse {
                qwen38_assert_k_complete_cols(body.cols)?;
                let name = qwen38_binary_matvec_kernel(body.cols);
                let grid = if body.cols > 2048 {
                    simd8_grid(body.rows)
                } else {
                    tg256_grid(body.rows)
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), |enc| {
                    self.encode_binary_args(enc, body, input, output)
                })
            } else {
                tcb.dispatch_threads(
                    qwen38_binary_family_kernel(self.decode_family),
                    (body.rows, 1, 1),
                    (256, 1, 1),
                    |enc| self.encode_binary_args(enc, body, input, output),
                )
            }
        }

        fn dispatch_residual(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuResidual,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if self.recon_fuse {
                qwen38_assert_k_complete_cols(body.binary.cols)?;
                let name = qwen38_residual_matvec_kernel(body.binary.cols);
                let grid = if body.binary.cols > 2048 {
                    simd8_grid(body.binary.rows)
                } else {
                    tg256_grid(body.binary.rows)
                };
                tcb.dispatch_threads(name, grid, (256, 1, 1), |enc| {
                    self.encode_binary_csr_args(enc, body, input, output)
                })
            } else {
                tcb.dispatch_threads(
                    qwen38_binary_family_kernel(self.decode_family),
                    (body.binary.rows, 1, 1),
                    (256, 1, 1),
                    |enc| self.encode_binary_args(enc, &body.binary, input, output),
                )?;
                tcb.dispatch_threads(
                    "q80_sparse_q1_apply_csr",
                    (body.binary.rows, 1, 1),
                    (256, 1, 1),
                    |enc| self.encode_csr_args(enc, body, input, output),
                )
            }
        }

        fn dispatch_factor(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            codes: &PinnedBuffer,
            scales: &PinnedBuffer,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            rows: u32,
            cols: u32,
            group_size: u32,
            bits: u32,
            bound: u32,
        ) -> Result<()> {
            let encode = |enc: &metal::ComputeCommandEncoderRef| {
                self.encode_factor_args(
                    enc, codes, scales, input, output, rows, cols, group_size, bits, bound,
                )
            };
            if self.recon_fuse {
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
                tcb.dispatch_threads(
                    qwen38_hgravs_family_kernel(self.decode_family),
                    (rows, 1, 1),
                    (256, 1, 1),
                    encode,
                )
            }
        }

        fn dispatch_hgravs(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuHgravs,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if body.right_rows as usize > QWEN38_MIXED_HGRAVS_RANK {
                return Err(mixed_error(format!(
                    "hgravs mid rank {} exceeds workspace {}",
                    body.right_rows, QWEN38_MIXED_HGRAVS_RANK
                )));
            }
            self.dispatch_factor(
                tcb,
                &body.right_codes,
                &body.right_scales,
                input,
                &self.workspace.hgravs_mid,
                body.right_rows,
                body.right_cols,
                body.group_size,
                body.bits,
                body.bound,
            )?;
            self.dispatch_factor(
                tcb,
                &body.left_codes,
                &body.left_scales,
                &self.workspace.hgravs_mid,
                output,
                body.left_rows,
                body.left_cols,
                body.group_size,
                body.bits,
                body.bound,
            )
        }

        fn dispatch_uniform(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            body: &GpuUniform,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
        ) -> Result<()> {
            if let Some((name, grid, tg)) = qwen38_hgravu01_geo_tpr64_launch_with_recon_fuse(
                body.bits,
                body.group_size,
                body.rows,
                body.cols,
                self.recon_fuse,
            ) {
                return tcb.dispatch_threads(name, grid, tg, |enc| {
                    self.encode_factor_args(
                        enc,
                        &body.codes,
                        &body.scales,
                        input,
                        output,
                        body.rows,
                        body.cols,
                        body.group_size,
                        body.bits,
                        body.bound,
                    )
                });
            }
            self.dispatch_factor(
                tcb,
                &body.codes,
                &body.scales,
                input,
                output,
                body.rows,
                body.cols,
                body.group_size,
                body.bits,
                body.bound,
            )
        }

        fn encode_fuse_qkvz(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            qkv: &PinnedBuffer,
            z: &PinnedBuffer,
            fused: &PinnedBuffer,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let n = layout.qkvz_rows() as u32;
            let kh = layout.key_heads as u32;
            let vpk = layout.values_per_key as u32;
            let kd = layout.key_head_dim as u32;
            let vd = layout.value_head_dim as u32;
            tcb.dispatch_threads("qwen38_fuse_split_qkvz_f32", (n, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(qkv), 0);
                enc.set_buffer(1, Some(z), 0);
                enc.set_buffer(2, Some(fused), 0);
                set_u32(enc, 3, kh);
                set_u32(enc, 4, vpk);
                set_u32(enc, 5, kd);
                set_u32(enc, 6, vd);
            })
        }

        fn encode_fuse_ba(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            b: &PinnedBuffer,
            a: &PinnedBuffer,
            fused: &PinnedBuffer,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let n = layout.ba_rows() as u32;
            let kh = layout.key_heads as u32;
            let vpk = layout.values_per_key as u32;
            tcb.dispatch_threads("qwen38_fuse_split_ba_f32", (n, 1, 1), (32, 1, 1), |enc| {
                enc.set_buffer(0, Some(b), 0);
                enc.set_buffer(1, Some(a), 0);
                enc.set_buffer(2, Some(fused), 0);
                set_u32(enc, 3, kh);
                set_u32(enc, 4, vpk);
            })
        }

        fn encode_split_deltanet_projections(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "linear_attn.in_proj_qkv.weight"),
                &self.workspace.normalized,
                &self.workspace.split_qkv,
            )?;
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "linear_attn.in_proj_z.weight"),
                &self.workspace.normalized,
                &self.workspace.z,
            )?;
            self.encode_fuse_qkvz(
                tcb,
                &self.workspace.split_qkv,
                &self.workspace.z,
                &self.workspace.qkvz,
            )?;
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "linear_attn.in_proj_b.weight"),
                &self.workspace.normalized,
                &self.workspace.split_b,
            )?;
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "linear_attn.in_proj_a.weight"),
                &self.workspace.normalized,
                &self.workspace.split_a,
            )?;
            self.encode_fuse_ba(
                tcb,
                &self.workspace.split_b,
                &self.workspace.split_a,
                &self.workspace.ba,
            )
        }

        fn has_weight(&self, name: &str) -> bool {
            if !self.weights.mixed.is_empty() {
                self.weights.mixed.contains_key(name) || self.weights.q4.contains_key(name)
            } else {
                self.weights.q4.contains_key(name) || self.weights.mixed.contains_key(name)
            }
        }

        fn encode_q4_matvec_kernel(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            kernel: &str,
        ) -> Result<()> {
            let weight = self.q4(name)?;
            self.encode_q4_matvec_weight(tcb, name, weight, input, output, kernel)
        }

        fn encode_q4_matvec_weight(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            weight: &Q4Weight,
            input: &PinnedBuffer,
            output: &PinnedBuffer,
            kernel: &str,
        ) -> Result<()> {
            self.record_q4_weight(weight);
            let rows = weight.rows as u32;
            let cols = weight.cols as u32;
            if weight.group_size == UNIFORM_Q4_GROUP_SIZE_128 {
                let (kname, grid, tg) = qwen38_uniform_q4_geo_tpr64_launch(128, rows, cols)
                    .ok_or_else(|| {
                        Error::Model(format!(
                            "{name} HQ30UQ4 group_size=128 does not bind geo_tpr64 (cols={cols})"
                        ))
                    })?;
                return tcb.dispatch_threads(kname, grid, tg, |encoder| {
                    encoder.set_buffer(0, Some(&weight.codes), 0);
                    encoder.set_buffer(1, Some(&weight.scales), 0);
                    encoder.set_buffer(2, Some(input), 0);
                    encoder.set_buffer(3, Some(output), 0);
                    encoder.set_bytes(4, 4, &rows as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &cols as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &weight.groups_per_row as *const u32 as *const _);
                });
            }
            let (grid, tg) = self.matvec_kernel.launch(rows);
            tcb.dispatch_threads(kernel, grid, tg, |encoder| {
                encoder.set_buffer(0, Some(&weight.codes), 0);
                encoder.set_buffer(1, Some(&weight.scales), 0);
                encoder.set_buffer(2, Some(input), 0);
                encoder.set_buffer(3, Some(output), 0);
                encoder.set_bytes(4, 4, &rows as *const u32 as *const _);
                encoder.set_bytes(5, 4, &cols as *const u32 as *const _);
                encoder.set_bytes(6, 4, &weight.groups_per_row as *const u32 as *const _);
            })
        }

        fn encode_independent_q4_pair(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            a_name: &str,
            a_input: &PinnedBuffer,
            a_output: &PinnedBuffer,
            b_name: &str,
            b_input: &PinnedBuffer,
            b_output: &PinnedBuffer,
        ) -> Result<()> {
            if self.concurrent_independent {
                tcb.begin_concurrent_group()?;
            }
            self.encode_q4_matvec(tcb, a_name, a_input, a_output)?;
            self.encode_q4_matvec(tcb, b_name, b_input, b_output)?;
            if self.concurrent_independent {
                tcb.end_concurrent_group()?;
            }
            Ok(())
        }

        fn encode_fused_affine_gate_up(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            gate: &GpuAffine,
            up: &GpuAffine,
            with_swiglu: bool,
        ) -> Result<()> {
            if gate.rows != up.rows || gate.cols != up.cols {
                return Err(mixed_error(format!(
                    "qwen38 fused affine gate/up shape mismatch: {}x{} vs {}x{}",
                    gate.rows, gate.cols, up.rows, up.cols
                )));
            }
            if gate.group_size != 64 || up.group_size != 64 || gate.bits != 2 || up.bits != 2 {
                return Err(mixed_error(format!(
                    "qwen38 fused affine gate/up refuses bits={}/{} group={}/{} (need 2 @ 64)",
                    gate.bits, up.bits, gate.group_size, up.group_size
                )));
            }
            let gate_b = gate.biases.as_ref().ok_or_else(|| {
                mixed_error("fused affine gate/up on a delta-only (q2f) tensor")
            })?;
            let up_b = up.biases.as_ref().ok_or_else(|| {
                mixed_error("fused affine gate/up on a delta-only (q2f) tensor")
            })?;
            self.record_affine_weight(gate);
            self.record_affine_weight(up);
            let rows = gate.rows;
            let cols = gate.cols;
            let (name, grid, tg) = qwen38_affine_gate_up_launch(self.affine2_geo, with_swiglu, rows);
            let xsum = self.affine2_geo.uses_xsum();
            if with_swiglu {
                tcb.dispatch_threads(name, grid, tg, |enc| {
                    enc.set_buffer(0, Some(&gate.codes), 0);
                    enc.set_buffer(1, Some(&gate.scales), 0);
                    enc.set_buffer(2, Some(gate_b), 0);
                    enc.set_buffer(3, Some(&up.codes), 0);
                    enc.set_buffer(4, Some(&up.scales), 0);
                    enc.set_buffer(5, Some(up_b), 0);
                    enc.set_buffer(6, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(7, Some(&self.workspace.act), 0);
                    if xsum {
                        enc.set_buffer(8, Some(&self.workspace.xsum64), 0);
                        set_u32(enc, 9, rows);
                        set_u32(enc, 10, cols);
                    } else {
                        set_u32(enc, 8, rows);
                        set_u32(enc, 9, cols);
                    }
                })
            } else {
                tcb.dispatch_threads(name, grid, tg, |enc| {
                    enc.set_buffer(0, Some(&gate.codes), 0);
                    enc.set_buffer(1, Some(&gate.scales), 0);
                    enc.set_buffer(2, Some(gate_b), 0);
                    enc.set_buffer(3, Some(&up.codes), 0);
                    enc.set_buffer(4, Some(&up.scales), 0);
                    enc.set_buffer(5, Some(up_b), 0);
                    enc.set_buffer(6, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(7, Some(&self.workspace.gate), 0);
                    enc.set_buffer(8, Some(&self.workspace.up), 0);
                    if xsum {
                        enc.set_buffer(9, Some(&self.workspace.xsum64), 0);
                        set_u32(enc, 10, rows);
                        set_u32(enc, 11, cols);
                    } else {
                        set_u32(enc, 9, rows);
                        set_u32(enc, 10, cols);
                    }
                })
            }
        }

        fn encode_fused_q2f_gate_up(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            gate: &GpuAffine,
            up: &GpuAffine,
            with_swiglu: bool,
        ) -> Result<()> {
            if gate.rows != up.rows || gate.cols != up.cols {
                return Err(mixed_error(format!(
                    "qwen38 fused q2f gate/up shape mismatch: {}x{} vs {}x{}",
                    gate.rows, gate.cols, up.rows, up.cols
                )));
            }
            if gate.group_size != 64 || up.group_size != 64 || gate.bits != 2 || up.bits != 2 {
                return Err(mixed_error(format!(
                    "qwen38 fused q2f gate/up refuses bits={}/{} group={}/{} (need 2 @ 64)",
                    gate.bits, up.bits, gate.group_size, up.group_size
                )));
            }
            self.record_affine_weight(gate);
            self.record_affine_weight(up);
            let rows = gate.rows;
            let cols = gate.cols;
            let (name, grid, tg) = qwen38_q2f_gate_up_launch(self.q2f_geo, with_swiglu, rows);
            if with_swiglu {
                tcb.dispatch_threads(name, grid, tg, |enc| {
                    enc.set_buffer(0, Some(&gate.codes), 0);
                    enc.set_buffer(1, Some(&gate.scales), 0);
                    enc.set_buffer(2, Some(&up.codes), 0);
                    enc.set_buffer(3, Some(&up.scales), 0);
                    enc.set_buffer(4, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(5, Some(&self.workspace.act), 0);
                    set_u32(enc, 6, rows);
                    set_u32(enc, 7, cols);
                })
            } else {
                tcb.dispatch_threads(name, grid, tg, |enc| {
                    enc.set_buffer(0, Some(&gate.codes), 0);
                    enc.set_buffer(1, Some(&gate.scales), 0);
                    enc.set_buffer(2, Some(&up.codes), 0);
                    enc.set_buffer(3, Some(&up.scales), 0);
                    enc.set_buffer(4, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(5, Some(&self.workspace.gate), 0);
                    enc.set_buffer(6, Some(&self.workspace.up), 0);
                    set_u32(enc, 7, rows);
                    set_u32(enc, 8, cols);
                })
            }
        }

        fn encode_fused_gate_up(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            with_swiglu: bool,
        ) -> Result<()> {
            let gate_name = self.layer_name(layer, "mlp.gate_proj.weight");
            let up_name = self.layer_name(layer, "mlp.up_proj.weight");
            if let (Some(gate), Some(up)) = (self.affine(&gate_name), self.affine(&up_name)) {
                if gate.biases.is_none() && up.biases.is_none() {
                    return self.encode_fused_q2f_gate_up(tcb, gate, up, with_swiglu);
                }
                return self.encode_fused_affine_gate_up(tcb, gate, up, with_swiglu);
            }
            let gate = self.q4(&gate_name)?;
            let up = self.q4(&up_name)?;
            if gate.rows != up.rows || gate.cols != up.cols {
                return Err(Error::Model(format!(
                    "qwen38 fused gate/up shape mismatch layer {layer}: {}x{} vs {}x{}",
                    gate.rows, gate.cols, up.rows, up.cols
                )));
            }
            if gate.group_size != UNIFORM_Q4_GROUP_SIZE || up.group_size != UNIFORM_Q4_GROUP_SIZE {
                return Err(Error::Model(format!(
                    "qwen38 fused gate/up refuses group_size {}/{} (need {UNIFORM_Q4_GROUP_SIZE})",
                    gate.group_size, up.group_size
                )));
            }
            self.record_q4_weight(gate);
            self.record_q4_weight(up);
            let rows = gate.rows as u32;
            let cols = gate.cols as u32;
            let gpr = (gate.cols / UNIFORM_Q4_GROUP_SIZE) as u32;
            let (grid, tg) = Qwen38MatvecKernel::GeoTpr64Tg128.launch(rows);
            if with_swiglu {
                tcb.dispatch_threads(QWEN38_Q4_GATE_UP_SWIGLU_KERNEL, grid, tg, |enc| {
                    enc.set_buffer(0, Some(&gate.codes), 0);
                    enc.set_buffer(1, Some(&gate.scales), 0);
                    enc.set_buffer(2, Some(&up.codes), 0);
                    enc.set_buffer(3, Some(&up.scales), 0);
                    enc.set_buffer(4, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(5, Some(&self.workspace.act), 0);
                    set_u32(enc, 6, rows);
                    set_u32(enc, 7, cols);
                    set_u32(enc, 8, gpr);
                })
            } else {
                tcb.dispatch_threads(QWEN38_Q4_GATE_UP_KERNEL, grid, tg, |enc| {
                    enc.set_buffer(0, Some(&gate.codes), 0);
                    enc.set_buffer(1, Some(&gate.scales), 0);
                    enc.set_buffer(2, Some(&up.codes), 0);
                    enc.set_buffer(3, Some(&up.scales), 0);
                    enc.set_buffer(4, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(5, Some(&self.workspace.gate), 0);
                    enc.set_buffer(6, Some(&self.workspace.up), 0);
                    set_u32(enc, 7, rows);
                    set_u32(enc, 8, cols);
                    set_u32(enc, 9, gpr);
                })
            }
        }

        fn encode_fused_pair_concat(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            a_name: &str,
            a_out: &PinnedBuffer,
            b_name: &str,
            b_out: &PinnedBuffer,
        ) -> Result<()> {
            let a = self.q4(a_name)?;
            let b = self.q4(b_name)?;
            if a.cols != b.cols {
                return Err(Error::Model(format!(
                    "qwen38 pair-concat col mismatch {a_name} {} vs {b_name} {}",
                    a.cols, b.cols
                )));
            }
            if a.group_size != UNIFORM_Q4_GROUP_SIZE || b.group_size != UNIFORM_Q4_GROUP_SIZE {
                return Err(Error::Model(format!(
                    "qwen38 pair-concat refuses group_size {}/{}",
                    a.group_size, b.group_size
                )));
            }
            self.record_q4_weight(a);
            self.record_q4_weight(b);
            let a_rows = a.rows as u32;
            let b_rows = b.rows as u32;
            let cols = a.cols as u32;
            let gpr = (a.cols / UNIFORM_Q4_GROUP_SIZE) as u32;
            let total = a_rows.saturating_add(b_rows);
            let (grid, tg) = Qwen38MatvecKernel::GeoTpr64Tg128.launch(total);
            tcb.dispatch_threads(qwen38_q4_kernel(QWEN38_Q4_PAIR_CONCAT_KERNEL), grid, tg, |enc| {
                enc.set_buffer(0, Some(&a.codes), 0);
                enc.set_buffer(1, Some(&a.scales), 0);
                enc.set_buffer(2, Some(&b.codes), 0);
                enc.set_buffer(3, Some(&b.scales), 0);
                enc.set_buffer(4, Some(&self.workspace.normalized), 0);
                enc.set_buffer(5, Some(a_out), 0);
                enc.set_buffer(6, Some(b_out), 0);
                set_u32(enc, 7, a_rows);
                set_u32(enc, 8, b_rows);
                set_u32(enc, 9, cols);
                set_u32(enc, 10, gpr);
            })
        }

        fn encode_fused_qkv(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let q = self.q4(self.layer_name(layer, "self_attn.q_proj.weight"))?;
            let k = self.q4(self.layer_name(layer, "self_attn.k_proj.weight"))?;
            let v = self.q4(self.layer_name(layer, "self_attn.v_proj.weight"))?;
            if q.cols != k.cols || q.cols != v.cols {
                return Err(Error::Model(format!(
                    "qwen38 fused QKV col mismatch layer {layer}"
                )));
            }
            if q.group_size != UNIFORM_Q4_GROUP_SIZE
                || k.group_size != UNIFORM_Q4_GROUP_SIZE
                || v.group_size != UNIFORM_Q4_GROUP_SIZE
            {
                return Err(Error::Model(format!(
                    "qwen38 fused QKV refuses group_size {}/{}/{}",
                    q.group_size, k.group_size, v.group_size
                )));
            }
            self.record_q4_weight(q);
            self.record_q4_weight(k);
            self.record_q4_weight(v);
            let q_rows = q.rows as u32;
            let k_rows = k.rows as u32;
            let v_rows = v.rows as u32;
            let cols = q.cols as u32;
            let gpr = (q.cols / UNIFORM_Q4_GROUP_SIZE) as u32;
            let total = q_rows.saturating_add(k_rows).saturating_add(v_rows);
            let (grid, tg) = Qwen38MatvecKernel::GeoTpr64Tg128.launch(total);
            tcb.dispatch_threads(qwen38_q4_kernel(QWEN38_Q4_QKV_GEO_KERNEL), grid, tg, |enc| {
                enc.set_buffer(0, Some(&q.codes), 0);
                enc.set_buffer(1, Some(&q.scales), 0);
                enc.set_buffer(2, Some(&k.codes), 0);
                enc.set_buffer(3, Some(&k.scales), 0);
                enc.set_buffer(4, Some(&v.codes), 0);
                enc.set_buffer(5, Some(&v.scales), 0);
                enc.set_buffer(6, Some(&self.workspace.normalized), 0);
                enc.set_buffer(7, Some(&self.workspace.q_proj), 0);
                enc.set_buffer(8, Some(&self.workspace.k_proj), 0);
                enc.set_buffer(9, Some(&self.workspace.v_proj), 0);
                set_u32(enc, 10, q_rows);
                set_u32(enc, 11, k_rows);
                set_u32(enc, 12, v_rows);
                set_u32(enc, 13, cols);
                set_u32(enc, 14, gpr);
            })
        }

        fn encode_fused_q2f_qkv(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let q_name = self.layer_name(layer, "self_attn.q_proj.weight");
            let k_name = self.layer_name(layer, "self_attn.k_proj.weight");
            let v_name = self.layer_name(layer, "self_attn.v_proj.weight");
            let q = self.affine(&q_name).ok_or_else(|| {
                mixed_error(format!("qwen38 fused Q2F QKV missing {q_name}"))
            })?;
            let k = self.affine(&k_name).ok_or_else(|| {
                mixed_error(format!("qwen38 fused Q2F QKV missing {k_name}"))
            })?;
            let v = self.affine(&v_name).ok_or_else(|| {
                mixed_error(format!("qwen38 fused Q2F QKV missing {v_name}"))
            })?;
            if q.rows == 0 || k.rows == 0 || v.rows == 0 || q.cols != k.cols || q.cols != v.cols {
                return Err(mixed_error(format!(
                    "qwen38 fused Q2F QKV shape mismatch layer {layer}: {}x{}, {}x{}, {}x{}",
                    q.rows, q.cols, k.rows, k.cols, v.rows, v.cols
                )));
            }
            if q.group_size != 64
                || k.group_size != 64
                || v.group_size != 64
                || q.bits != 2
                || k.bits != 2
                || v.bits != 2
                || q.biases.is_some()
                || k.biases.is_some()
                || v.biases.is_some()
                || q.cols % 64 != 0
            {
                return Err(mixed_error(format!(
                    "qwen38 fused Q2F QKV refuses bits={}/{}/{} group={}/{}/{} bias={}/{}/{} cols={}",
                    q.bits,
                    k.bits,
                    v.bits,
                    q.group_size,
                    k.group_size,
                    v.group_size,
                    q.biases.is_some(),
                    k.biases.is_some(),
                    v.biases.is_some(),
                    q.cols
                )));
            }
            self.record_affine_weight(q);
            self.record_affine_weight(k);
            self.record_affine_weight(v);
            let q_rows = q.rows;
            let k_rows = k.rows;
            let v_rows = v.rows;
            let cols = q.cols;
            let total = q_rows.saturating_add(k_rows).saturating_add(v_rows);
            let grid = total.div_ceil(2).saturating_mul(128).max(128);
            tcb.dispatch_threads(
                QWEN38_Q2F_QKV_GEO_KERNEL,
                (grid, 1, 1),
                (128, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q.codes), 0);
                    enc.set_buffer(1, Some(&q.scales), 0);
                    enc.set_buffer(2, Some(&k.codes), 0);
                    enc.set_buffer(3, Some(&k.scales), 0);
                    enc.set_buffer(4, Some(&v.codes), 0);
                    enc.set_buffer(5, Some(&v.scales), 0);
                    enc.set_buffer(6, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(7, Some(&self.workspace.q_proj), 0);
                    enc.set_buffer(8, Some(&self.workspace.k_proj), 0);
                    enc.set_buffer(9, Some(&self.workspace.v_proj), 0);
                    set_u32(enc, 10, q_rows);
                    set_u32(enc, 11, k_rows);
                    set_u32(enc, 12, v_rows);
                    set_u32(enc, 13, cols);
                },
            )
        }

        fn encode_fused_q2f_pair(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let qkvz_name = self.layer_name(layer, "linear_attn.in_proj_qkvz.weight");
            let ba_name = self.layer_name(layer, "linear_attn.in_proj_ba.weight");
            let qkvz = self.affine(&qkvz_name).ok_or_else(|| {
                mixed_error(format!("qwen38 fused Q2F QKVZ/BA missing {qkvz_name}"))
            })?;
            let ba = self.affine(&ba_name).ok_or_else(|| {
                mixed_error(format!("qwen38 fused Q2F QKVZ/BA missing {ba_name}"))
            })?;
            if qkvz.rows == 0
                || ba.rows == 0
                || qkvz.cols != ba.cols
                || qkvz.cols % 64 != 0
                || [qkvz, ba].iter().any(|body| {
                    body.bits != 2 || body.group_size != 64 || body.biases.is_some()
                })
            {
                return Err(mixed_error(format!(
                    "qwen38 fused Q2F QKVZ/BA refuses shapes {}x{} and {}x{} or non-Q2F codec",
                    qkvz.rows, qkvz.cols, ba.rows, ba.cols
                )));
            }
            self.record_affine_weight(qkvz);
            self.record_affine_weight(ba);
            let left_rows = qkvz.rows;
            let right_rows = ba.rows;
            let cols = qkvz.cols;
            let total = left_rows.saturating_add(right_rows);
            let grid = total.div_ceil(2).saturating_mul(128).max(128);
            tcb.dispatch_threads(
                QWEN38_Q2F_PAIR_GEO_KERNEL,
                (grid, 1, 1),
                (128, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&qkvz.codes), 0);
                    enc.set_buffer(1, Some(&qkvz.scales), 0);
                    enc.set_buffer(2, Some(&ba.codes), 0);
                    enc.set_buffer(3, Some(&ba.scales), 0);
                    enc.set_buffer(4, Some(&self.workspace.normalized), 0);
                    enc.set_buffer(5, Some(&self.workspace.qkvz), 0);
                    enc.set_buffer(6, Some(&self.workspace.ba), 0);
                    set_u32(enc, 7, left_rows);
                    set_u32(enc, 8, right_rows);
                    set_u32(enc, 9, cols);
                },
            )
        }

        fn timed_cb(
            &self,
            encode: impl FnOnce(&mut TokenCommandBuffer<'_>) -> Result<()>,
        ) -> Result<CommandBufferTiming> {
            let mut tcb = TokenCommandBuffer::new(&self.context);
            encode(&mut tcb)?;
            tcb.commit_and_wait_timed()
        }

        fn encode_gated_delta(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            rec_off: u64,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let heads = layout.value_heads as u32;
            let kd = layout.key_head_dim as u32;
            let vd = layout.value_head_dim as u32;
            let (kernel, grid) = if self.deltanet_vi_parallel {
                // Both 128-element reductions in the vi kernel run on thread 0
                // while 127 lanes wait. HAWKING_DN_VI_SIMD=0 restores it; the
                // simd sibling is not bit-identical (tree vs serial
                // association) and is gated on greedy token identity.
                (
                    if self.deltanet_vi_simd {
                        "qwen38_gated_delta_decode_vi_simd"
                    } else {
                        "qwen38_gated_delta_decode_vi"
                    },
                    (kd, heads, vd),
                )
            } else {
                (
                    "qwen80_gated_delta_decode_tg",
                    (kd, heads, 1),
                )
            };
            tcb.dispatch_threads(kernel, grid, (kd, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&self.workspace.rec_state), rec_off);
                encoder.set_buffer(1, Some(&self.workspace.repeated_q), 0);
                encoder.set_buffer(2, Some(&self.workspace.repeated_k), 0);
                encoder.set_buffer(3, Some(&self.workspace.conv_v), 0);
                encoder.set_buffer(4, Some(&self.workspace.decay), 0);
                encoder.set_buffer(5, Some(&self.workspace.beta), 0);
                encoder.set_buffer(6, Some(&self.workspace.rec_out), 0);
                encoder.set_bytes(7, 4, &heads as *const u32 as *const _);
                encoder.set_bytes(8, 4, &kd as *const u32 as *const _);
                encoder.set_bytes(9, 4, &vd as *const u32 as *const _);
                encoder.set_threadgroup_memory_length(0, 128 * 4);
            })
        }

        fn encode_gated_delta_fused_ba(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            rec_off: u64,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let heads = layout.value_heads as u32;
            let kd = layout.key_head_dim as u32;
            let vd = layout.value_head_dim as u32;
            let a_log = self.f32(self.layer_name(layer, "linear_attn.A_log"))?;
            let dt_bias = self.f32(self.layer_name(layer, "linear_attn.dt_bias"))?;
            let kernel = self.dn_state_kernel.fused_ba_name(self.fuse_ba_delta_bad);
            let (grid_z, tg_bytes) = if self.fuse_ba_delta_bad {
                (vd, 512u64)
            } else {
                match self.dn_state_kernel {
                    Qwen38DeltaNetStateKernel::WidenF4 => (vd / 4, 512u64),
                    Qwen38DeltaNetStateKernel::CoalesceTg32 => (vd / 32, QWEN38_DN_STATE_TG32_BYTES),
                    Qwen38DeltaNetStateKernel::Baseline => (vd, 512u64),
                }
            };
            tcb.dispatch_threads(kernel, (kd, heads, grid_z), (kd, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&self.workspace.rec_state), rec_off);
                encoder.set_buffer(1, Some(&self.workspace.repeated_q), 0);
                encoder.set_buffer(2, Some(&self.workspace.repeated_k), 0);
                encoder.set_buffer(3, Some(&self.workspace.conv_v), 0);
                encoder.set_buffer(4, Some(&self.workspace.ba), 0);
                encoder.set_buffer(5, Some(a_log), 0);
                encoder.set_buffer(6, Some(dt_bias), 0);
                encoder.set_buffer(7, Some(&self.workspace.rec_out), 0);
                encoder.set_bytes(8, 4, &heads as *const u32 as *const _);
                encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                encoder.set_threadgroup_memory_length(0, tg_bytes);
            })
        }

        fn encode_dn_ba_and_delta(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            rec_off: u64,
        ) -> Result<()> {
            // WidenF4 / CoalesceTg32 are fused-ba siblings: they consume
            // projected_ba in-register. Without this branch,
            // HAWKING_QWEN38_DN_STATE=widen_f4 was a no-op on the
            // production 628 graph (unfused vi-SIMD still launched).
            if qwen38_dn_state_uses_fused_ba(self.fuse_ba_delta, self.dn_state_kernel) {
                self.encode_gated_delta_fused_ba(tcb, rec_off, layer)
            } else {
                self.encode_ba_to_decay(tcb, layer)?;
                self.encode_gated_delta(tcb, rec_off)
            }
        }

        fn encode_mixer(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            match self.mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => self.encode_deltanet(tcb, layer),
                Qwen38MixerKind::Gqa => self.encode_gqa(tcb, layer),
            }
        }

        fn encode_mixer_gemvs_only(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_mixer_gemvs_only_mixed(tcb, layer);
            }
            match self.mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => {
                    self.encode_independent_q4_pair(
                        tcb,
                        self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                        &self.workspace.normalized,
                        &self.workspace.qkvz,
                        self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                        &self.workspace.normalized,
                        &self.workspace.ba,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        self.layer_name(layer, "linear_attn.out_proj.weight"),
                        &self.workspace.gated,
                        &self.workspace.mixer,
                    )
                }
                Qwen38MixerKind::Gqa => {
                    if self.concurrent_independent {
                        tcb.begin_concurrent_group()?;
                    }
                    self.encode_q4_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.q_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.q_proj,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.k_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.k_proj,
                    )?;
                    self.encode_q4_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.v_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.v_proj,
                    )?;
                    if self.concurrent_independent {
                        tcb.end_concurrent_group()?;
                    }
                    self.encode_q4_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.o_proj.weight"),
                        &self.workspace.gated_attn,
                        &self.workspace.mixer,
                    )
                }
            }
        }

        fn encode_mlp_matvecs_only(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_mlp_matvecs_only_mixed(tcb, layer);
            }
            self.encode_independent_q4_pair(
                tcb,
                self.layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
                self.layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            self.encode_q4_matvec(
                tcb,
                self.layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )
        }

        fn workspace_f32<'a>(&'a self, which: &str) -> Result<&'a PinnedBuffer> {
            match which {
                "gate" => Ok(&self.workspace.gate),
                "up" => Ok(&self.workspace.up),
                "act" => Ok(&self.workspace.act),
                "down" => Ok(&self.workspace.down),
                "hidden" => Ok(&self.workspace.hidden),
                "normalized" => Ok(&self.workspace.normalized),
                "logits" => Ok(&self.workspace.logits),
                "mixer" => Ok(&self.workspace.mixer),
                "q_proj" => Ok(&self.workspace.q_proj),
                "k_proj" => Ok(&self.workspace.k_proj),
                "v_proj" => Ok(&self.workspace.v_proj),
                "qkvz" => Ok(&self.workspace.qkvz),
                "ba" => Ok(&self.workspace.ba),
                // The DeltaNet carry. Addressable so a prefix's recurrent state
                // can be snapshotted and restored; see prefix_checkpoint.
                "rec_state" => Ok(&self.workspace.rec_state),
                "conv_state" => Ok(&self.workspace.conv_state),
                other => Err(Error::Model(format!(
                    "qwen38 unknown workspace buffer {other}"
                ))),
            }
        }

        pub fn read_f32_workspace(&self, which: &str, n: usize) -> Result<Vec<f32>> {
            let buffer = self.workspace_f32(which)?;
            let bytes = n
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Model("qwen38 read overflow".into()))?;
            if buffer.length() < bytes as u64 {
                return Err(Error::Model(format!(
                    "qwen38 {which} is {} bytes, need {bytes}",
                    buffer.length()
                )));
            }
            let mut out = vec![0.0f32; n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.contents() as *const f32,
                    out.as_mut_ptr(),
                    n,
                );
            }
            Ok(out)
        }

        pub fn write_f32_workspace(&self, which: &str, values: &[f32]) -> Result<()> {
            let buffer = self.workspace_f32(which)?;
            let bytes = values
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| Error::Model("qwen38 write overflow".into()))?;
            if buffer.length() < bytes as u64 {
                return Err(Error::Model(format!(
                    "qwen38 {which} is {} bytes, need {bytes}",
                    buffer.length()
                )));
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buffer.contents() as *mut f32,
                    values.len(),
                );
            }
            Ok(())
        }

        pub fn apply_fusion(
            &mut self,
            mlp: Qwen38MlpFusion,
            fuse_gqa_qkv: bool,
            fuse_dn_inproj: bool,
        ) {
            self.mlp_fusion = mlp;
            self.fuse_gqa_qkv = fuse_gqa_qkv;
            self.fuse_dn_inproj = fuse_dn_inproj;
        }

        pub fn apply_affine2_geo(&mut self, geo: Affine2Geo) {
            self.affine2_geo = geo;
            // Preserve the pre-existing API contract: callers that selected a
            // single geometry before Q2F got its independent environment
            // override still retarget both packed 2-bit families.
            self.q2f_geo = geo;
        }

        pub fn apply_q2f_geo(&mut self, geo: Affine2Geo) {
            self.q2f_geo = geo;
        }

        pub fn set_fuse_add_rmsnorm(&mut self, on: bool, bad: bool) {
            self.fuse_add_rmsnorm = on;
            self.fuse_add_rmsnorm_bad = bad;
        }

        pub fn set_fuse_attention_gate(&mut self, on: bool) {
            self.fuse_attention_gate = on;
        }

        pub fn set_fuse_ba_delta(&mut self, on: bool, bad: bool) {
            self.fuse_ba_delta = on;
            self.fuse_ba_delta_bad = bad;
        }

        pub fn set_dn_state_kernel(&mut self, kernel: Qwen38DeltaNetStateKernel) {
            self.dn_state_kernel = kernel;
        }

        /// Record a dense-W reconstruct. Production packed GEMV never calls this.
        pub fn account_dense_w(&mut self, n: u64) {
            self.dense_w_materialized += n;
        }

        pub fn set_q2f_force_serial(&mut self, on: bool) {
            self.q2f_force_serial = on;
        }

        /// Cover the token graph with one serial compute encoder.
        /// Forces `concurrent_independent` off so a nested concurrent group
        /// cannot fight the open serial encoder.
        pub fn set_serial_token_encoder(&mut self, on: bool) {
            self.serial_token_encoder = on;
            if on {
                self.concurrent_independent = false;
            }
        }

        fn encode_full_token(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            token: u32,
        ) -> Result<()> {
            if self.serial_token_encoder {
                tcb.begin_serial_group()?;
            }
            self.encode_embed(tcb, token)?;
            self.encode_layers(tcb)?;
            self.encode_terminal(tcb)?;
            if self.serial_token_encoder {
                tcb.end_serial_group()?;
            }
            Ok(())
        }

        pub fn theoretical_dispatches(&self) -> u64 {
            let n = qwen38_fused_dispatches_per_token_full(
                self.mlp_fusion,
                self.fuse_gqa_qkv,
                self.fuse_dn_inproj,
                self.fuse_add_rmsnorm,
                qwen38_dn_state_uses_fused_ba(self.fuse_ba_delta, self.dn_state_kernel),
            );
            n.saturating_sub(if self.fuse_attention_gate {
                QWEN38_ATTENTION_GATE_SAVED_PER_TOKEN
            } else {
                0
            })
        }

        /// Kernel name `encode_dn_ba_and_delta` will actually dispatch.
        /// Used by the 628-graph A/B to prove production launched f4
        /// rather than measuring a probe beside it.
        pub fn launched_gated_delta_kernel(&self) -> &'static str {
            if qwen38_dn_state_uses_fused_ba(self.fuse_ba_delta, self.dn_state_kernel) {
                self.dn_state_kernel.fused_ba_name(self.fuse_ba_delta_bad)
            } else if std::env::var("HAWKING_DN_VI_SIMD")
                .map(|v| v != "0")
                .unwrap_or(true)
            {
                "qwen38_gated_delta_decode_vi_simd"
            } else {
                "qwen38_gated_delta_decode_vi"
            }
        }

        /// Encode one complete token and return the TCB dispatch count.
        /// Mutates recurrent/KV state — caller should `reset` around a probe.
        pub fn measure_token_dispatches(&mut self, token: u32) -> Result<(u32, u64, CommandBufferTiming)> {
            let (sampled, timing) = self.step(token)?;
            Ok((sampled, timing.dispatches, timing))
        }

        pub fn measure_isolated_dense_mlp(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
                }
                Ok(())
            })
        }

        fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }

        /// Fused gate+up(+SwiGLU) vs the two-matvec + SwiGLU path on a known x.
        /// Does not materialize a dense parent W.
        pub fn measure_mlp_fusion_parity(&self, layer: usize) -> Result<Qwen38FusionParity> {
            let n_hidden = QWEN38_HIDDEN;
            let n_mid = QWEN38_INTERMEDIATE;
            let mut x = vec![0.0f32; n_hidden];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 17) as f32) * 0.01 - 0.08;
            }
            self.write_f32_workspace("normalized", &x)?;

            let unfused = self.timed_cb(|tcb| {
                self.encode_named_matvec(
                    tcb,
                    self.layer_name(layer, "mlp.gate_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.gate,
                )?;
                self.encode_named_matvec(
                    tcb,
                    self.layer_name(layer, "mlp.up_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.up,
                )?;
                let n = n_mid as u32;
                tcb.dispatch_threads(
                    qwen38_swiglu_family_kernel(self.decode_family),
                    (n, 1, 1),
                    (n.min(256).max(1), 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                        encoder.set_buffer(1, Some(&self.workspace.up), 0);
                        encoder.set_buffer(2, Some(&self.workspace.act), 0);
                        encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                    },
                )
            })?;
            let gate_u = self.read_f32_workspace("gate", n_mid)?;
            let up_u = self.read_f32_workspace("up", n_mid)?;
            let act_u = self.read_f32_workspace("act", n_mid)?;

            let pair = self.timed_cb(|tcb| self.encode_fused_gate_up(tcb, layer, false))?;
            let gate_p = self.read_f32_workspace("gate", n_mid)?;
            let up_p = self.read_f32_workspace("up", n_mid)?;

            let swiglu = self.timed_cb(|tcb| self.encode_fused_gate_up(tcb, layer, true))?;
            let act_s = self.read_f32_workspace("act", n_mid)?;

            Ok(Qwen38FusionParity {
                fusion: "gate_up_swiglu",
                layer,
                unfused_dispatches: unfused.dispatches,
                fused_pair_dispatches: pair.dispatches,
                fused_swiglu_dispatches: swiglu.dispatches,
                unfused_gpu_ns: unfused.gpu_ns,
                fused_pair_gpu_ns: pair.gpu_ns,
                fused_swiglu_gpu_ns: swiglu.gpu_ns,
                max_abs_diff_gate: Self::max_abs_diff(&gate_u, &gate_p),
                max_abs_diff_up: Self::max_abs_diff(&up_u, &up_p),
                max_abs_diff_act: Self::max_abs_diff(&act_u, &act_s),
                dense_w_materialized: 0,
            })
        }

        pub fn measure_qkv_fusion_parity(&self, layer: usize) -> Result<Qwen38FusionParity> {
            let n_hidden = QWEN38_HIDDEN;
            let q_n = QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM * 2;
            let kv_n = QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let mut x = vec![0.0f32; n_hidden];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 13) as f32) * 0.02 - 0.12;
            }
            self.write_f32_workspace("normalized", &x)?;
            let unfused = self.timed_cb(|tcb| {
                self.encode_q4_matvec(
                    tcb,
                    self.layer_name(layer, "self_attn.q_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.q_proj,
                )?;
                self.encode_q4_matvec(
                    tcb,
                    self.layer_name(layer, "self_attn.k_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.k_proj,
                )?;
                self.encode_q4_matvec(
                    tcb,
                    self.layer_name(layer, "self_attn.v_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.v_proj,
                )
            })?;
            let q_u = self.read_f32_workspace("q_proj", q_n)?;
            let k_u = self.read_f32_workspace("k_proj", kv_n)?;
            let v_u = self.read_f32_workspace("v_proj", kv_n)?;
            let fused = self.timed_cb(|tcb| self.encode_fused_qkv(tcb, layer))?;
            let q_f = self.read_f32_workspace("q_proj", q_n)?;
            let k_f = self.read_f32_workspace("k_proj", kv_n)?;
            let v_f = self.read_f32_workspace("v_proj", kv_n)?;
            Ok(Qwen38FusionParity {
                fusion: "gqa_qkv",
                layer,
                unfused_dispatches: unfused.dispatches,
                fused_pair_dispatches: fused.dispatches,
                fused_swiglu_dispatches: fused.dispatches,
                unfused_gpu_ns: unfused.gpu_ns,
                fused_pair_gpu_ns: fused.gpu_ns,
                fused_swiglu_gpu_ns: fused.gpu_ns,
                max_abs_diff_gate: Self::max_abs_diff(&q_u, &q_f),
                max_abs_diff_up: Self::max_abs_diff(&k_u, &k_f),
                max_abs_diff_act: Self::max_abs_diff(&v_u, &v_f),
                dense_w_materialized: 0,
            })
        }

        pub fn measure_dn_inproj_fusion_parity(&self, layer: usize) -> Result<Qwen38FusionParity> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let n_hidden = QWEN38_HIDDEN;
            let mut x = vec![0.0f32; n_hidden];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i % 11) as f32) * 0.015 - 0.07;
            }
            self.write_f32_workspace("normalized", &x)?;
            let unfused = self.timed_cb(|tcb| {
                self.encode_independent_q4_pair(
                    tcb,
                    self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                    &self.workspace.normalized,
                    &self.workspace.qkvz,
                    self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                    &self.workspace.normalized,
                    &self.workspace.ba,
                )
            })?;
            let qkvz_u = self.read_f32_workspace("qkvz", layout.qkvz_rows())?;
            let ba_u = self.read_f32_workspace("ba", layout.ba_rows())?;
            let fused = self.timed_cb(|tcb| {
                self.encode_fused_pair_concat(
                    tcb,
                    self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                    &self.workspace.qkvz,
                    self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                    &self.workspace.ba,
                )
            })?;
            let qkvz_f = self.read_f32_workspace("qkvz", layout.qkvz_rows())?;
            let ba_f = self.read_f32_workspace("ba", layout.ba_rows())?;
            Ok(Qwen38FusionParity {
                fusion: "dn_qkvz_ba",
                layer,
                unfused_dispatches: unfused.dispatches,
                fused_pair_dispatches: fused.dispatches,
                fused_swiglu_dispatches: fused.dispatches,
                unfused_gpu_ns: unfused.gpu_ns,
                fused_pair_gpu_ns: fused.gpu_ns,
                fused_swiglu_gpu_ns: fused.gpu_ns,
                max_abs_diff_gate: Self::max_abs_diff(&qkvz_u, &qkvz_f),
                max_abs_diff_up: Self::max_abs_diff(&ba_u, &ba_f),
                max_abs_diff_act: 0.0,
                dense_w_materialized: 0,
            })
        }

        /// Residual add + RMSNorm vs the two-dispatch production pair.
        /// `bad=true` binds the plain-weight kernel (intentionally diverges).
        pub fn measure_add_rmsnorm_fusion_parity(
            &mut self,
            layer: usize,
            bad: bool,
        ) -> Result<Qwen38FusionParity> {
            let n = QWEN38_HIDDEN;
            let mut residual = vec![0.0f32; n];
            let mut delta = vec![0.0f32; n];
            for i in 0..n {
                residual[i] = ((i % 19) as f32) * 0.02 - 0.17;
                delta[i] = ((i % 13) as f32) * 0.015 - 0.09;
            }
            self.write_f32_workspace("hidden", &residual)?;
            self.write_f32_workspace("mixer", &delta)?;
            let weight_name = self
                .layer_name(layer, "post_attention_layernorm.weight")
                .to_owned();
            let saved_bad = self.fuse_add_rmsnorm_bad;
            self.fuse_add_rmsnorm_bad = false;
            let unfused = self.timed_cb(|tcb| {
                qwen_next_add_residual_tcb(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    QWEN38_HIDDEN,
                )?;
                self.encode_rmsnorm(
                    tcb,
                    &self.workspace.first_residual,
                    &weight_name,
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )
            })?;
            let residual_u = {
                let buf = &self.workspace.first_residual;
                let mut out = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
                }
                out
            };
            let norm_u = self.read_f32_workspace("normalized", n)?;
            self.write_f32_workspace("hidden", &residual)?;
            self.write_f32_workspace("mixer", &delta)?;
            self.fuse_add_rmsnorm_bad = bad;
            let fused = self.timed_cb(|tcb| {
                self.encode_add_residual_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    &weight_name,
                    &self.workspace.normalized,
                )
            })?;
            let residual_f = {
                let buf = &self.workspace.first_residual;
                let mut out = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
                }
                out
            };
            let norm_f = self.read_f32_workspace("normalized", n)?;
            self.fuse_add_rmsnorm_bad = saved_bad;
            Ok(Qwen38FusionParity {
                fusion: if bad {
                    "add_residual_rmsnorm_plainweight"
                } else {
                    "add_residual_rmsnorm"
                },
                layer,
                unfused_dispatches: unfused.dispatches,
                fused_pair_dispatches: fused.dispatches,
                fused_swiglu_dispatches: fused.dispatches,
                unfused_gpu_ns: unfused.gpu_ns,
                fused_pair_gpu_ns: fused.gpu_ns,
                fused_swiglu_gpu_ns: fused.gpu_ns,
                max_abs_diff_gate: Self::max_abs_diff(&residual_u, &residual_f),
                max_abs_diff_up: Self::max_abs_diff(&norm_u, &norm_f),
                max_abs_diff_act: Self::max_abs_diff(&norm_u, &norm_f),
                dense_w_materialized: 0,
            })
        }

        /// ba_to_decay + gated-delta vs the fused kernel on layer 0's real A_log/dt_bias.
        /// `bad=true` binds the identity-decay control (must diverge).
        pub fn measure_ba_delta_fusion_parity(
            &mut self,
            layer: usize,
            bad: bool,
        ) -> Result<Qwen38FusionParity> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = self.deltanet_state_slot(layer)?;
            let rec_n = layout.recurrent_state_elements();
            let rec_off = (slot * rec_n * 4) as u64;
            let ba_n = layout.ba_rows();
            let q_n = layout.key_heads * layout.key_head_dim;
            let v_n = layout.value_elements();
            let fill = |n: usize, modulus: usize, scale: f32, bias: f32| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i % modulus) as f32) * scale - bias)
                    .collect()
            };
            let write_buf = |buf: &PinnedBuffer, values: &[f32]| {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        values.as_ptr(),
                        buf.contents() as *mut f32,
                        values.len(),
                    );
                }
            };
            let read_buf = |buf: &PinnedBuffer, n: usize| -> Vec<f32> {
                let mut out = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
                }
                out
            };
            let ba = fill(ba_n, 11, 0.02, 0.1);
            let q = fill(q_n, 13, 0.015, 0.08);
            let k = fill(q_n, 17, 0.018, 0.09);
            let v = fill(v_n, 19, 0.012, 0.07);
            let rec = fill(rec_n, 23, 0.004, 0.05);
            write_buf(&self.workspace.ba, &ba);
            write_buf(&self.workspace.repeated_q, &q);
            write_buf(&self.workspace.repeated_k, &k);
            write_buf(&self.workspace.conv_v, &v);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rec.as_ptr(),
                    (self.workspace.rec_state.contents() as *mut u8).add(rec_off as usize)
                        as *mut f32,
                    rec_n,
                );
            }
            let saved_on = self.fuse_ba_delta;
            let saved_bad = self.fuse_ba_delta_bad;
            self.fuse_ba_delta = false;
            self.fuse_ba_delta_bad = false;
            let unfused = self.timed_cb(|tcb| {
                self.encode_ba_to_decay(tcb, layer)?;
                self.encode_gated_delta(tcb, rec_off)
            })?;
            let rec_out_u = read_buf(&self.workspace.rec_out, v_n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rec.as_ptr(),
                    (self.workspace.rec_state.contents() as *mut u8).add(rec_off as usize)
                        as *mut f32,
                    rec_n,
                );
            }
            self.fuse_ba_delta = true;
            self.fuse_ba_delta_bad = bad;
            let fused = self.timed_cb(|tcb| self.encode_gated_delta_fused_ba(tcb, rec_off, layer))?;
            let rec_out_f = read_buf(&self.workspace.rec_out, v_n);
            self.fuse_ba_delta = saved_on;
            self.fuse_ba_delta_bad = saved_bad;
            let diff = Self::max_abs_diff(&rec_out_u, &rec_out_f);
            Ok(Qwen38FusionParity {
                fusion: if bad {
                    "ba_delta_identity"
                } else {
                    "ba_delta"
                },
                layer,
                unfused_dispatches: unfused.dispatches,
                fused_pair_dispatches: fused.dispatches,
                fused_swiglu_dispatches: fused.dispatches,
                unfused_gpu_ns: unfused.gpu_ns,
                fused_pair_gpu_ns: fused.gpu_ns,
                fused_swiglu_gpu_ns: fused.gpu_ns,
                max_abs_diff_gate: diff,
                max_abs_diff_up: diff,
                max_abs_diff_act: diff,
                dense_w_materialized: 0,
            })
        }

        /// Candidate gated-delta kernel vs N025 `vi_simd_ba` on layer 0.
        /// Compares rec_out AND the recurrent state (not just the readout).
        pub fn measure_dn_state_kernel_parity(
            &mut self,
            layer: usize,
            candidate: Qwen38DeltaNetStateKernel,
            bad: bool,
        ) -> Result<Qwen38DeltaNetStateParity> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = self.deltanet_state_slot(layer)?;
            let rec_n = layout.recurrent_state_elements();
            let rec_off = (slot * rec_n * 4) as u64;
            let ba_n = layout.ba_rows();
            let q_n = layout.key_heads * layout.key_head_dim;
            let v_n = layout.value_elements();
            let fill = |n: usize, modulus: usize, scale: f32, bias: f32| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i % modulus) as f32) * scale - bias)
                    .collect()
            };
            let write_buf = |buf: &PinnedBuffer, values: &[f32]| unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    buf.contents() as *mut f32,
                    values.len(),
                );
            };
            let read_buf = |buf: &PinnedBuffer, n: usize| -> Vec<f32> {
                let mut out = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(buf.contents() as *const f32, out.as_mut_ptr(), n);
                }
                out
            };
            let write_rec = |values: &[f32]| unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    (self.workspace.rec_state.contents() as *mut u8).add(rec_off as usize)
                        as *mut f32,
                    rec_n,
                );
            };
            let read_rec = || -> Vec<f32> {
                let mut out = vec![0.0f32; rec_n];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (self.workspace.rec_state.contents() as *const u8).add(rec_off as usize)
                            as *const f32,
                        out.as_mut_ptr(),
                        rec_n,
                    );
                }
                out
            };
            let ba = fill(ba_n, 11, 0.02, 0.1);
            let q = fill(q_n, 13, 0.015, 0.08);
            let k = fill(q_n, 17, 0.018, 0.09);
            let v = fill(v_n, 19, 0.012, 0.07);
            let rec = fill(rec_n, 23, 0.004, 0.05);
            write_buf(&self.workspace.ba, &ba);
            write_buf(&self.workspace.repeated_q, &q);
            write_buf(&self.workspace.repeated_k, &k);
            write_buf(&self.workspace.conv_v, &v);
            write_rec(&rec);
            let saved_on = self.fuse_ba_delta;
            let saved_bad = self.fuse_ba_delta_bad;
            let saved_k = self.dn_state_kernel;
            self.fuse_ba_delta = true;
            self.fuse_ba_delta_bad = false;
            self.dn_state_kernel = Qwen38DeltaNetStateKernel::Baseline;
            let base = self.timed_cb(|tcb| self.encode_gated_delta_fused_ba(tcb, rec_off, layer))?;
            let rec_out_b = read_buf(&self.workspace.rec_out, v_n);
            let rec_state_b = read_rec();
            write_rec(&rec);
            self.dn_state_kernel = candidate;
            self.fuse_ba_delta_bad = bad;
            let cand = self.timed_cb(|tcb| self.encode_gated_delta_fused_ba(tcb, rec_off, layer))?;
            let rec_out_c = read_buf(&self.workspace.rec_out, v_n);
            let rec_state_c = read_rec();
            self.fuse_ba_delta = saved_on;
            self.fuse_ba_delta_bad = saved_bad;
            self.dn_state_kernel = saved_k;
            Ok(Qwen38DeltaNetStateParity {
                kernel: candidate.fused_ba_name(bad),
                layer,
                max_abs_diff_rec_out: Self::max_abs_diff(&rec_out_b, &rec_out_c),
                max_abs_diff_rec_state: Self::max_abs_diff(&rec_state_b, &rec_state_c),
                baseline_gpu_ns: base.gpu_ns,
                candidate_gpu_ns: cand.gpu_ns,
                baseline_dispatches: base.dispatches,
                candidate_dispatches: cand.dispatches,
                dense_w_materialized: 0,
            })
        }

        pub fn measure_isolated_dn_state_update(&self) -> Result<CommandBufferTiming> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    if self.mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                        continue;
                    }
                    let slot = self.deltanet_state_slot(layer)?;
                    let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
                    self.encode_dn_ba_and_delta(tcb, layer, rec_off)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_dn_inproj(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    if self.mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                        continue;
                    }
                    let fused_qkvz = self.layer_name(layer, "linear_attn.in_proj_qkvz.weight");
                    let fused_ba = self.layer_name(layer, "linear_attn.in_proj_ba.weight");
                    if self.has_weight(&fused_qkvz) && self.has_weight(&fused_ba) {
                        if self.fuse_dn_inproj
                            && self.weights.q4.contains_key(fused_qkvz)
                            && self.weights.q4.contains_key(fused_ba)
                        {
                            self.encode_fused_pair_concat(
                                tcb,
                                &fused_qkvz,
                                &self.workspace.qkvz,
                                &fused_ba,
                                &self.workspace.ba,
                            )?;
                        } else if self.fuse_dn_inproj && self.can_fuse_q2f_pair(layer) {
                            self.encode_fused_q2f_pair(tcb, layer)?;
                        } else {
                            self.encode_named_matvec(
                                tcb,
                                &fused_qkvz,
                                &self.workspace.normalized,
                                &self.workspace.qkvz,
                            )?;
                            self.encode_named_matvec(
                                tcb,
                                &fused_ba,
                                &self.workspace.normalized,
                                &self.workspace.ba,
                            )?;
                        }
                    }
                }
                Ok(())
            })
        }

        fn encode_organ_gqa_compute(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let slot = self.gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = slot * slot_elems * 4;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                self.layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            let q_name = self.layer_name(layer, "self_attn.q_proj.weight");
            let k_name = self.layer_name(layer, "self_attn.k_proj.weight");
            let v_name = self.layer_name(layer, "self_attn.v_proj.weight");
            if self.fuse_gqa_qkv
                && self.weights.q4.contains_key(q_name)
                && self.weights.q4.contains_key(k_name)
                && self.weights.q4.contains_key(v_name)
            {
                self.encode_fused_qkv(tcb, layer)?;
            } else if self.fuse_gqa_qkv && self.can_fuse_q2f_qkv(layer) {
                self.encode_fused_q2f_qkv(tcb, layer)?;
            } else {
                self.encode_named_matvec(
                    tcb,
                    &q_name,
                    &self.workspace.normalized,
                    &self.workspace.q_proj,
                )?;
                self.encode_named_matvec(
                    tcb,
                    &k_name,
                    &self.workspace.normalized,
                    &self.workspace.k_proj,
                )?;
                self.encode_named_matvec(
                    tcb,
                    &v_name,
                    &self.workspace.normalized,
                    &self.workspace.v_proj,
                )?;
            }
            self.encode_rope_cache(tcb, layer)?;
            let seq = self.position.max(1).min(self.max_seq_len);
            self.encode_gqa_mha_and_gate(tcb, cache_off as usize, seq)?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        fn encode_organ_dn_compute(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = self.deltanet_state_slot(layer)?;
            let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
            self.encode_rmsnorm(
                tcb,
                &self.workspace.hidden,
                self.layer_name(layer, "input_layernorm.weight"),
                &self.workspace.normalized,
                QWEN38_HIDDEN as u32,
            )?;
            let fused_qkvz = self.layer_name(layer, "linear_attn.in_proj_qkvz.weight");
            let fused_ba = self.layer_name(layer, "linear_attn.in_proj_ba.weight");
            if self.has_weight(&fused_qkvz) && self.has_weight(&fused_ba) {
                if self.fuse_dn_inproj
                    && self.weights.q4.contains_key(fused_qkvz)
                    && self.weights.q4.contains_key(fused_ba)
                {
                    self.encode_fused_pair_concat(
                        tcb,
                        &fused_qkvz,
                        &self.workspace.qkvz,
                        &fused_ba,
                        &self.workspace.ba,
                    )?;
                } else if self.fuse_dn_inproj && self.can_fuse_q2f_pair(layer) {
                    self.encode_fused_q2f_pair(tcb, layer)?;
                } else {
                    self.encode_named_matvec(
                        tcb,
                        &fused_qkvz,
                        &self.workspace.normalized,
                        &self.workspace.qkvz,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        &fused_ba,
                        &self.workspace.normalized,
                        &self.workspace.ba,
                    )?;
                }
            } else if self.has_weight(self.layer_name(layer, "linear_attn.in_proj_qkv.weight"))
            {
                self.encode_split_deltanet_projections(tcb, layer)?;
            } else {
                return Err(Error::Model(format!(
                    "layer {layer} mixer projections missing for organ isolate"
                )));
            }
            self.encode_rearrange(tcb, layer)?;
            self.encode_dn_ba_and_delta(tcb, layer, rec_off)?;
            self.encode_gated_rmsnorm(tcb, layer)?;
            qwen_next_add_residual_tcb(
                tcb,
                &self.workspace.hidden,
                &self.workspace.mixer,
                &self.workspace.first_residual,
                QWEN38_HIDDEN,
            )
        }

        /// Isolated organ CBs: one family, all layers, GPUEnd−GPUStart.
        /// Production remains one CB; these partition it. Caller scales.
        pub fn measure_isolated_organ(&self, organ: &str) -> Result<CommandBufferTiming> {
            match organ {
                "embedding" => self.timed_cb(|tcb| self.encode_embed(tcb, 1)),
                "gqa_attention" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? == Qwen38MixerKind::Gqa {
                            self.encode_organ_gqa_compute(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "deltanet" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_organ_dn_compute(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "mlp_gate_up" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_rmsnorm(
                            tcb,
                            &self.workspace.first_residual,
                            self.layer_name(layer, "post_attention_layernorm.weight"),
                            &self.workspace.normalized,
                            QWEN38_HIDDEN as u32,
                        )?;
                        self.encode_fused_gate_up(tcb, layer, true)?;
                    }
                    Ok(())
                }),
                "mlp_down" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_named_matvec(
                            tcb,
                            self.layer_name(layer, "mlp.down_proj.weight"),
                            &self.workspace.act,
                            &self.workspace.down,
                        )?;
                        qwen_next_add_residual_tcb(
                            tcb,
                            &self.workspace.first_residual,
                            &self.workspace.down,
                            &self.workspace.hidden,
                            QWEN38_HIDDEN,
                        )?;
                    }
                    Ok(())
                }),
                "q4_remainder" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        match self.mixer_kind(layer)? {
                            Qwen38MixerKind::DeltaNet => {
                                self.encode_named_matvec(
                                    tcb,
                                    self.layer_name(layer, "linear_attn.out_proj.weight"),
                                    &self.workspace.gated,
                                    &self.workspace.mixer,
                                )?;
                            }
                            Qwen38MixerKind::Gqa => {
                                self.encode_named_matvec(
                                    tcb,
                                    self.layer_name(layer, "self_attn.o_proj.weight"),
                                    &self.workspace.gated_attn,
                                    &self.workspace.mixer,
                                )?;
                            }
                        }
                    }
                    Ok(())
                }),
                "lm_head" => self.timed_cb(|tcb| {
                    self.encode_rmsnorm(
                        tcb,
                        &self.workspace.hidden,
                        "language_model.model.norm.weight",
                        &self.workspace.normalized,
                        QWEN38_HIDDEN as u32,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        "language_model.lm_head.weight",
                        &self.workspace.normalized,
                        &self.workspace.logits,
                    )
                }),
                "sampling" => self.timed_cb(|tcb| self.encode_argmax(tcb)),
                "noop_empty" => self.timed_cb(|_tcb| Ok(())),
                other => Err(Error::Model(format!(
                    "qwen38 unknown isolated organ {other}"
                ))),
            }
        }

        pub fn measure_named_matvec(&self, name: &str, output: &str) -> Result<CommandBufferTiming> {
            match output {
                "gate" | "up" | "down" | "logits" | "mixer" | "qkvz" | "hidden" => {}
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown matvec output {other}"
                    )))
                }
            }
            self.timed_cb(|tcb| {
                let out_buf = match output {
                    "gate" => &self.workspace.gate,
                    "up" => &self.workspace.up,
                    "down" => &self.workspace.down,
                    "logits" => &self.workspace.logits,
                    "mixer" => &self.workspace.mixer,
                    "qkvz" => &self.workspace.qkvz,
                    _ => &self.workspace.hidden,
                };
                self.encode_named_matvec(tcb, name, &self.workspace.normalized, out_buf)
            })
        }

        pub fn measure_isolated_mlp_full(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mlp_matvecs(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_mlp_matvecs_only(tcb, layer)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mlp_one_proj(&self, which: &str) -> Result<CommandBufferTiming> {
            let suffix = match which {
                "gate" => "mlp.gate_proj.weight",
                "up" => "mlp.up_proj.weight",
                "down" => "mlp.down_proj.weight",
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 mlp proj {other} is not gate/up/down"
                    )))
                }
            };
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    let (input, output) = match which {
                        "gate" => (&self.workspace.normalized, &self.workspace.gate),
                        "up" => (&self.workspace.normalized, &self.workspace.up),
                        _ => (&self.workspace.act, &self.workspace.down),
                    };
                    self.encode_q4_matvec(
                        tcb,
                        self.layer_name(layer, suffix),
                        input,
                        output,
                    )?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_gated_delta(&self) -> Result<CommandBufferTiming> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    if self.mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                        continue;
                    }
                    let slot = self.deltanet_state_slot(layer)?;
                    let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
                    self.encode_gated_delta(tcb, rec_off)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_mixer_gemvs(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    self.encode_mixer_gemvs_only(tcb, layer)?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_lm_head(&self) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| {
                self.encode_q4_matvec(
                    tcb,
                    "language_model.lm_head.weight",
                    &self.workspace.normalized,
                    &self.workspace.logits,
                )
            })
        }

        pub fn measure_isolated_embed(&self, token: u32) -> Result<CommandBufferTiming> {
            self.timed_cb(|tcb| self.encode_embed(tcb, token))
        }

        pub fn alloc_profile_buffer(&self, bytes: usize) -> Result<PinnedBuffer> {
            self.context.new_buffer_checked(bytes)
        }

        /// Everything needed to resume decoding as if a token prefix had just
        /// been stepped: the DeltaNet carry and the sequence position.
        ///
        /// KV is deliberately NOT captured. It is indexed by position, so for a
        /// prefix of identical tokens the entries at 0..position are already the
        /// same bytes whichever request wrote them. The recurrent state is the
        /// only part that cannot be reconstructed by indexing, because it is a
        /// running summary with no per-position addressing.
        ///
        /// This is EXACT, not an approximation: the carry after tokens 0..N is a
        /// function of those tokens alone, so two prompts sharing a prefix have
        /// identical state at N by construction.
        pub fn prefix_checkpoint(&self) -> Result<Qwen38PrefixCheckpoint> {
            Ok(Qwen38PrefixCheckpoint {
                position: self.position,
                rec_state: self.read_f32_workspace("rec_state", self.rec_state_f32_count())?,
                conv_state: self
                    .read_f32_workspace("conv_state", self.conv_state_f32_count())?,
            })
        }

        /// Resume from a checkpoint. The caller MUST have verified that the
        /// prompt begins with exactly the tokens the checkpoint was taken over;
        /// restoring against a different prefix conditions generation on tokens
        /// that are not in the prompt, and nothing downstream could detect it.
        pub fn restore_prefix(&mut self, checkpoint: &Qwen38PrefixCheckpoint) -> Result<()> {
            if checkpoint.rec_state.len() != self.rec_state_f32_count()
                || checkpoint.conv_state.len() != self.conv_state_f32_count()
            {
                return Err(Error::Model(
                    "qwen38 prefix checkpoint was taken on a different geometry".into(),
                ));
            }
            if checkpoint.position > self.max_seq_len {
                return Err(Error::Model(
                    "qwen38 prefix checkpoint position exceeds max_seq_len".into(),
                ));
            }
            self.write_f32_workspace("rec_state", &checkpoint.rec_state)?;
            self.write_f32_workspace("conv_state", &checkpoint.conv_state)?;
            self.position = checkpoint.position;
            Ok(())
        }

        pub fn rec_state_f32_count(&self) -> usize {
            (self.workspace.rec_state.length() as usize) / 4
        }

        pub fn conv_state_f32_count(&self) -> usize {
            (self.workspace.conv_state.length() as usize) / 4
        }

        pub fn gqa_cache_f32_count(&self) -> usize {
            (self.workspace.gqa_key.length() as usize
                + self.workspace.gqa_value.length() as usize)
                / 4
        }

        fn encode_silu(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            let n = QWEN38_INTERMEDIATE as u32;
            tcb.dispatch_threads(
                qwen38_swiglu_family_kernel(self.decode_family),
                (n, 1, 1),
                (n.min(256).max(1), 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                    encoder.set_buffer(1, Some(&self.workspace.up), 0);
                    encoder.set_buffer(2, Some(&self.workspace.act), 0);
                    encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                },
            )
        }

        fn encode_rearrange(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = self.deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let conv_w = self.f32(self.layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )
        }

        fn encode_ba_to_decay(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let a_log = self.f32(self.layer_name(layer, "linear_attn.A_log"))?;
            let dt_bias = self.f32(self.layer_name(layer, "linear_attn.dt_bias"))?;
            tcb.dispatch_threads(
                "qwen80_ba_to_decay_beta_f32",
                (layout.value_heads as u32, 1, 1),
                (16, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.ba), 0);
                    encoder.set_buffer(1, Some(a_log), 0);
                    encoder.set_buffer(2, Some(dt_bias), 0);
                    encoder.set_buffer(3, Some(&self.workspace.decay), 0);
                    encoder.set_buffer(4, Some(&self.workspace.beta), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    encoder.set_bytes(5, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &vpk as *const u32 as *const _);
                },
            )
        }

        fn encode_gated_rmsnorm(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let norm_w = self.f32(self.layer_name(layer, "linear_attn.norm.weight"))?;
            let dn_tg = self.dn_rmsnorm_tg;
            let (dn_name, dn_grid, dn_tgd) = if dn_tg > 0 {
                ("qwen80_deltanet_gated_rmsnorm_tg",
                 (layout.value_heads as u32 * dn_tg, 1, 1), (dn_tg, 1, 1))
            } else {
                ("qwen80_deltanet_gated_rmsnorm_f32",
                 (layout.value_heads as u32, 1, 1), (16, 1, 1))
            };
            tcb.dispatch_threads(
                dn_name,
                dn_grid,
                dn_tgd,
                |encoder| {
                    if dn_tg > 0 {
                        encoder.set_threadgroup_memory_length(0, (dn_tg as u64) * 4);
                    }
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )
        }

        fn encode_rope_cache(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let slot = self.gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            let q_norm = self.f32(self.layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(self.layer_name(layer, "self_attn.k_norm.weight"))?;
            let rope_tg = self.rope_tg;
            let (rope_name, rope_grid, rope_tgd) = if rope_tg > 0 {
                ("qwen38_gqa_qk_norm_rope_cache_tg",
                 (QWEN38_GQA_HEADS as u32 * rope_tg, 1, 1), (rope_tg, 1, 1))
            } else {
                ("qwen38_gqa_qk_norm_rope_cache_f32",
                 (QWEN38_GQA_HEADS as u32, 1, 1), (QWEN38_GQA_HEADS as u32, 1, 1))
            };
            tcb.dispatch_threads(
                rope_name,
                rope_grid,
                rope_tgd,
                |encoder| {
                    if rope_tg > 0 {
                        encoder.set_threadgroup_memory_length(0, (rope_tg as u64) * 4);
                    }
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position.saturating_sub(1).min(self.max_seq_len - 1) as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )
        }

        fn encode_mha(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            let slot = self.gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            let seq = self.position.max(1).min(self.max_seq_len);
            mha_decode_f32_tcb(
                tcb,
                &self.workspace.query,
                &self.workspace.gqa_key,
                cache_off as usize,
                &self.workspace.gqa_value,
                cache_off as usize,
                &self.workspace.attn,
                seq,
                QWEN38_GQA_HEAD_DIM,
                QWEN38_GQA_HEADS,
                QWEN38_GQA_KV_HEADS,
            )
        }

        /// Encode the Qwen3.8 GQA attention result in the buffer consumed by
        /// o_proj. The fused candidate keeps the generic MHA reduction but
        /// applies the q-projection sigmoid gate in the same kernel, removing
        /// one device dispatch and one full hidden-width intermediate write
        /// per GQA layer. The explicit unfused branch remains the diagnostic
        /// and A/B authority path.
        fn encode_gqa_mha_and_gate(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            cache_off: usize,
            seq_len: usize,
        ) -> Result<()> {
            if self.fuse_attention_gate {
                return Err(mixed_error(
                    "qwen38 gated MHA (mha_decode_f32_qwen38_gated_tcb) is not \
                     in this commit's kernels; leave HAWKING_QWEN38_FUSE_ATTENTION_GATE off",
                ));
            } else {
                mha_decode_f32_tcb(
                    tcb,
                    &self.workspace.query,
                    &self.workspace.gqa_key,
                    cache_off,
                    &self.workspace.gqa_value,
                    cache_off,
                    &self.workspace.attn,
                    seq_len,
                    QWEN38_GQA_HEAD_DIM,
                    QWEN38_GQA_HEADS,
                    QWEN38_GQA_KV_HEADS,
                )?;
                self.encode_sigmoid_gate(tcb)
            }
        }

        fn encode_sigmoid_gate(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            let query_dim = (QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u32;
            let head_dim = QWEN38_GQA_HEAD_DIM as u32;
            tcb.dispatch_threads(
                "qwen38_attention_apply_sigmoid_gate",
                (query_dim, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.attn), 0);
                    encoder.set_buffer(1, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.gated_attn), 0);
                    encoder.set_bytes(3, 4, &query_dim as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &head_dim as *const u32 as *const _);
                },
            )
        }

        fn encode_argmax(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            // The single-threadgroup argmax scans the whole vocabulary on one of
            // 60 cores, and the two-pass form below is 26x faster in isolation
            // (0.3395 -> 0.0131 ms) with token-identical output. It is DEFAULT
            // OFF anyway, because none of that saving reaches the token: four
            // paired runs put the end-to-end median at -0.045 ms, i.e. nothing,
            // against a within-arm spread of 0.281 ms. Shipping a second
            // dispatch and two buffers for an unmeasurable win is not worth it.
            // HAWKING_ARGMAX_TWO_PASS=1 enables it; see
            // receipts/ascent-2026-08-16/ARGMAX_TWO_PASS_NO_TRANSFER.json.
            if !self.argmax_two_pass {
                return sample_argmax_f32_tcb(
                    tcb,
                    &self.workspace.logits,
                    &self.workspace.sampled,
                    QWEN38_VOCAB,
                );
            }
            let vocab = QWEN38_VOCAB as u32;
            let groups = ARGMAX_GROUPS as u32;
            tcb.dispatch_threads(
                "sample_argmax_f32_pass1",
                (groups * 256, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&self.workspace.logits), 0);
                    enc.set_buffer(1, Some(&self.workspace.argmax_part_v), 0);
                    enc.set_buffer(2, Some(&self.workspace.argmax_part_i), 0);
                    enc.set_bytes(3, 4, &vocab as *const u32 as *const _);
                    enc.set_threadgroup_memory_length(0, 256 * 4);
                    enc.set_threadgroup_memory_length(1, 256 * 4);
                },
            )?;
            tcb.dispatch_threads(
                "sample_argmax_f32_pass2",
                (256, 1, 1),
                (256, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&self.workspace.argmax_part_v), 0);
                    enc.set_buffer(1, Some(&self.workspace.argmax_part_i), 0);
                    enc.set_buffer(2, Some(&self.workspace.sampled), 0);
                    enc.set_bytes(3, 4, &groups as *const u32 as *const _);
                    enc.set_threadgroup_memory_length(0, 256 * 4);
                    enc.set_threadgroup_memory_length(1, 256 * 4);
                },
            )
        }

        fn encode_f32_stream(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            src: &PinnedBuffer,
            dst: &PinnedBuffer,
            n_f32: u32,
        ) -> Result<()> {
            tcb.dispatch_threads(
                QWEN38_F32_STREAM_PROBE_KERNEL,
                (n_f32, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(src), 0);
                    encoder.set_buffer(1, Some(dst), 0);
                    encoder.set_bytes(2, 4, &n_f32 as *const u32 as *const _);
                },
            )
        }

        pub fn measure_isolated_family(
            &self,
            family: &str,
        ) -> Result<CommandBufferTiming> {
            match family {
                "input_norms" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_rmsnorm(
                            tcb,
                            &self.workspace.hidden,
                            self.layer_name(layer, "input_layernorm.weight"),
                            &self.workspace.normalized,
                            QWEN38_HIDDEN as u32,
                        )?;
                    }
                    Ok(())
                }),
                "post_norms" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_rmsnorm(
                            tcb,
                            &self.workspace.first_residual,
                            self.layer_name(layer, "post_attention_layernorm.weight"),
                            &self.workspace.normalized,
                            QWEN38_HIDDEN as u32,
                        )?;
                    }
                    Ok(())
                }),
                "final_norm" => self.timed_cb(|tcb| {
                    self.encode_rmsnorm(
                        tcb,
                        &self.workspace.hidden,
                        "language_model.model.norm.weight",
                        &self.workspace.normalized,
                        QWEN38_HIDDEN as u32,
                    )
                }),
                "silu_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        self.encode_silu(tcb)?;
                    }
                    Ok(())
                }),
                "mlp_residual_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        qwen_next_add_residual_tcb(
                            tcb,
                            &self.workspace.first_residual,
                            &self.workspace.down,
                            &self.workspace.hidden,
                            QWEN38_HIDDEN,
                        )?;
                    }
                    Ok(())
                }),
                "mixer_residual_64" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_LAYERS {
                        qwen_next_add_residual_tcb(
                            tcb,
                            &self.workspace.hidden,
                            &self.workspace.mixer,
                            &self.workspace.first_residual,
                            QWEN38_HIDDEN,
                        )?;
                    }
                    Ok(())
                }),
                "rearrange_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_rearrange(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "ba_to_decay_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_ba_to_decay(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "gated_rmsnorm_48" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? == Qwen38MixerKind::DeltaNet {
                            self.encode_gated_rmsnorm(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "rope_cache_16" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? == Qwen38MixerKind::Gqa {
                            self.encode_rope_cache(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "mha_16" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? == Qwen38MixerKind::Gqa {
                            self.encode_mha(tcb, layer)?;
                        }
                    }
                    Ok(())
                }),
                "sigmoid_16" => self.timed_cb(|tcb| {
                    for _ in 0..QWEN38_GQA_LAYERS {
                        self.encode_sigmoid_gate(tcb)?;
                    }
                    Ok(())
                }),
                "argmax" => self.timed_cb(|tcb| self.encode_argmax(tcb)),
                "dn_gemvs" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                            continue;
                        }
                        self.encode_independent_q4_pair(
                            tcb,
                            self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                            &self.workspace.normalized,
                            &self.workspace.qkvz,
                            self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                            &self.workspace.normalized,
                            &self.workspace.ba,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            self.layer_name(layer, "linear_attn.out_proj.weight"),
                            &self.workspace.gated,
                            &self.workspace.mixer,
                        )?;
                    }
                    Ok(())
                }),
                "gqa_gemvs" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? != Qwen38MixerKind::Gqa {
                            continue;
                        }
                        self.encode_q4_matvec(
                            tcb,
                            self.layer_name(layer, "self_attn.q_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.q_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            self.layer_name(layer, "self_attn.k_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.k_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            self.layer_name(layer, "self_attn.v_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.v_proj,
                        )?;
                        self.encode_q4_matvec(
                            tcb,
                            self.layer_name(layer, "self_attn.o_proj.weight"),
                            &self.workspace.gated_attn,
                            &self.workspace.mixer,
                        )?;
                    }
                    Ok(())
                }),
                other => Err(Error::Model(format!(
                    "qwen38 unknown isolated family {other}"
                ))),
            }
        }

        pub fn measure_isolated_mlp_one_proj_kernel(
            &self,
            which: &str,
            kernel: &str,
        ) -> Result<CommandBufferTiming> {
            let suffix = match which {
                "gate" => "mlp.gate_proj.weight",
                "up" => "mlp.up_proj.weight",
                "down" => "mlp.down_proj.weight",
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 mlp proj {other} is not gate/up/down"
                    )))
                }
            };
            self.timed_cb(|tcb| {
                for layer in 0..QWEN38_LAYERS {
                    let (input, output) = match which {
                        "gate" => (&self.workspace.normalized, &self.workspace.gate),
                        "up" => (&self.workspace.normalized, &self.workspace.up),
                        _ => (&self.workspace.act, &self.workspace.down),
                    };
                    self.encode_q4_matvec_kernel(
                        tcb,
                        self.layer_name(layer, suffix),
                        input,
                        output,
                        kernel,
                    )?;
                }
                Ok(())
            })
        }

        pub fn measure_isolated_class_gemvs_kernel(
            &self,
            class: &str,
            kernel: &str,
        ) -> Result<CommandBufferTiming> {
            match class {
                "mlp" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "mlp.gate_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.gate,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "mlp.up_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.up,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "mlp.down_proj.weight"),
                            &self.workspace.act,
                            &self.workspace.down,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "dn" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? != Qwen38MixerKind::DeltaNet {
                            continue;
                        }
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                            &self.workspace.normalized,
                            &self.workspace.qkvz,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                            &self.workspace.normalized,
                            &self.workspace.ba,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "linear_attn.out_proj.weight"),
                            &self.workspace.gated,
                            &self.workspace.mixer,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "gqa" => self.timed_cb(|tcb| {
                    for layer in 0..QWEN38_LAYERS {
                        if self.mixer_kind(layer)? != Qwen38MixerKind::Gqa {
                            continue;
                        }
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "self_attn.q_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.q_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "self_attn.k_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.k_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "self_attn.v_proj.weight"),
                            &self.workspace.normalized,
                            &self.workspace.v_proj,
                            kernel,
                        )?;
                        self.encode_q4_matvec_kernel(
                            tcb,
                            self.layer_name(layer, "self_attn.o_proj.weight"),
                            &self.workspace.gated_attn,
                            &self.workspace.mixer,
                            kernel,
                        )?;
                    }
                    Ok(())
                }),
                "lm_head" => self.timed_cb(|tcb| {
                    self.encode_q4_matvec_kernel(
                        tcb,
                        "language_model.lm_head.weight",
                        &self.workspace.normalized,
                        &self.workspace.logits,
                        kernel,
                    )
                }),
                other => Err(Error::Model(format!(
                    "qwen38 unknown gemv class {other}"
                ))),
            }
        }

        pub fn measure_f32_stream(
            &self,
            which: &str,
            dest: &PinnedBuffer,
        ) -> Result<CommandBufferTiming> {
            let (src, n) = match which {
                "rec_state" => (&self.workspace.rec_state, self.rec_state_f32_count()),
                "conv_state" => (&self.workspace.conv_state, self.conv_state_f32_count()),
                "gqa_key" => (
                    &self.workspace.gqa_key,
                    (self.workspace.gqa_key.length() as usize) / 4,
                ),
                "gqa_value" => (
                    &self.workspace.gqa_value,
                    (self.workspace.gqa_value.length() as usize) / 4,
                ),
                other => {
                    return Err(Error::Model(format!(
                        "qwen38 unknown stream source {other}"
                    )))
                }
            };
            let n = n.min((dest.length() as usize) / 4);
            self.timed_cb(|tcb| self.encode_f32_stream(tcb, src, dest, n as u32))
        }

        pub fn step_decomposed(&mut self, token: u32) -> Result<(u32, Qwen38ClassTiming)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            let mut out = Qwen38ClassTiming::default();
            let embed = self.timed_cb(|tcb| self.encode_embed(tcb, token))?;
            out.embed_gpu_ns = embed.gpu_ns;
            out.embed_wait_ns = embed.wait_ns;
            for layer in 0..QWEN38_LAYERS {
                let mixer = self.timed_cb(|tcb| self.encode_mixer(tcb, layer))?;
                let mixer_gpu = mixer.gpu_ns.unwrap_or(0);
                out.mixer_gpu_ns = out.mixer_gpu_ns.saturating_add(mixer_gpu);
                out.mixer_wait_ns = out.mixer_wait_ns.saturating_add(mixer.wait_ns);
                out.layer_mixer_gpu_ns.push(mixer_gpu);
                match self.mixer_kind(layer)? {
                    Qwen38MixerKind::DeltaNet => {
                        out.deltanet_gpu_ns = out.deltanet_gpu_ns.saturating_add(mixer_gpu);
                    }
                    Qwen38MixerKind::Gqa => {
                        out.gqa_gpu_ns = out.gqa_gpu_ns.saturating_add(mixer_gpu);
                    }
                }
                let mlp = self.timed_cb(|tcb| {
                    self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)
                })?;
                let mlp_gpu = mlp.gpu_ns.unwrap_or(0);
                out.mlp_gpu_ns = out.mlp_gpu_ns.saturating_add(mlp_gpu);
                out.mlp_wait_ns = out.mlp_wait_ns.saturating_add(mlp.wait_ns);
                out.layer_mlp_gpu_ns.push(mlp_gpu);
            }
            let term = self.timed_cb(|tcb| self.encode_terminal(tcb))?;
            out.terminal_gpu_ns = term.gpu_ns;
            out.terminal_wait_ns = term.wait_ns;
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            self.position = self.position.saturating_add(1);
            out.sampled = sampled;
            Ok((sampled, out))
        }

        fn encode_rmsnorm(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            input: &PinnedBuffer,
            weight_name: &str,
            output: &PinnedBuffer,
            hidden: u32,
        ) -> Result<()> {
            let weight = self.f32(weight_name)?;
            let rms_tg = self.rmsnorm_tg;
            let xsum = self.affine2_geo.uses_xsum();
            let (rms_name, rms_n) = if xsum {
                (QWEN38_RMSNORM_XSUM64_KERNEL, if rms_tg > 0 { rms_tg } else { 256 })
            } else if rms_tg > 0 {
                ("qwen80_residual_rmsnorm_tg", rms_tg)
            } else {
                ("qwen80_residual_rmsnorm_f32", 256)
            };
            tcb.dispatch_threads(
                rms_name,
                (rms_n, 1, 1),
                (rms_n, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(input), 0);
                    encoder.set_buffer(1, Some(weight), 0);
                    encoder.set_buffer(2, Some(output), 0);
                    if xsum {
                        encoder.set_buffer(3, Some(&self.workspace.xsum64), 0);
                        encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                        encoder.set_bytes(5, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    } else {
                        encoder.set_bytes(3, 4, &hidden as *const u32 as *const _);
                        encoder.set_bytes(4, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    }
                    // sized to the ACTUAL threadgroup: the scratch is one float per thread and
                    // a hardcoded 256 silently under-allocates for any larger tg, which showed
                    // up immediately as diverged tokens rather than as a crash
                    encoder.set_threadgroup_memory_length(0, (rms_n as u64) * 4);
                },
            )
        }

        fn add_rmsnorm_kernel(&self) -> &'static str {
            if self.fuse_add_rmsnorm_bad {
                QWEN38_ADD_RMSNORM_BAD_KERNEL
            } else if self.affine2_geo.uses_xsum() {
                QWEN38_ADD_RMSNORM_XSUM64_KERNEL
            } else {
                QWEN38_ADD_RMSNORM_KERNEL
            }
        }

        fn next_norm_weight_name(&self, layer: usize) -> &str {
            if layer + 1 < QWEN38_LAYERS {
                self.layer_name(layer + 1, "input_layernorm.weight")
            } else {
                "language_model.model.norm.weight"
            }
        }

        fn encode_add_residual_rmsnorm(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            residual_in: &PinnedBuffer,
            delta: &PinnedBuffer,
            residual_out: &PinnedBuffer,
            weight_name: &str,
            x_norm: &PinnedBuffer,
        ) -> Result<()> {
            let weight = self.f32(weight_name)?;
            let rms_tg = self.rmsnorm_tg;
            let tg = if rms_tg > 0 { rms_tg } else { 256 };
            let hidden = QWEN38_HIDDEN as u32;
            let xsum = self.affine2_geo.uses_xsum() && !self.fuse_add_rmsnorm_bad;
            tcb.dispatch_threads(
                self.add_rmsnorm_kernel(),
                (tg, 1, 1),
                (tg, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(residual_in), 0);
                    encoder.set_buffer(1, Some(delta), 0);
                    encoder.set_buffer(2, Some(residual_out), 0);
                    encoder.set_buffer(3, Some(weight), 0);
                    encoder.set_buffer(4, Some(x_norm), 0);
                    if xsum {
                        encoder.set_buffer(5, Some(&self.workspace.xsum64), 0);
                        encoder.set_bytes(6, 4, &hidden as *const u32 as *const _);
                        encoder.set_bytes(7, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    } else {
                        encoder.set_bytes(5, 4, &hidden as *const u32 as *const _);
                        encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    }
                    encoder.set_threadgroup_memory_length(0, (tg as u64) * 4);
                },
            )
        }

        fn encode_embed(&self, tcb: &mut TokenCommandBuffer<'_>, token: u32) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_embed_mixed(tcb, token);
            }
            let weight = self.q4("language_model.model.embed_tokens.weight")?;
            if weight.rows != QWEN38_VOCAB || weight.cols != QWEN38_HIDDEN {
                return Err(Error::Model("qwen38 embed shape drifted".into()));
            }
            self.record_q4_embedding_row(weight);
            let hidden = QWEN38_HIDDEN as u32;
            let vocab = QWEN38_VOCAB as u32;
            let group = weight.group_size as u32;
            tcb.dispatch_threads(
                "qwen_uniform_q4_embedding_lookup",
                (hidden, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&weight.codes), 0);
                    encoder.set_buffer(1, Some(&weight.scales), 0);
                    encoder.set_buffer(2, Some(&self.workspace.hidden), 0);
                    encoder.set_bytes(3, 4, &token as *const u32 as *const _);
                    encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &vocab as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &group as *const u32 as *const _);
                },
            )
        }

        fn encode_dense_mlp(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            input: &PinnedBuffer,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_dense_mlp_mixed(tcb, layer, input);
            }
            let n = QWEN38_INTERMEDIATE as u32;
            if !self.fuse_add_rmsnorm {
                self.encode_rmsnorm(
                    tcb,
                    input,
                    self.layer_name(layer, "post_attention_layernorm.weight"),
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )?;
            }
            match self.mlp_fusion {
                Qwen38MlpFusion::Off => {
                    self.encode_independent_q4_pair(
                        tcb,
                        self.layer_name(layer, "mlp.gate_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.gate,
                        self.layer_name(layer, "mlp.up_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.up,
                    )?;
                    tcb.dispatch_threads(
                        qwen38_swiglu_family_kernel(self.decode_family),
                        (n, 1, 1),
                        (n.min(256).max(1), 1, 1),
                        |encoder| {
                            encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                            encoder.set_buffer(1, Some(&self.workspace.up), 0);
                            encoder.set_buffer(2, Some(&self.workspace.act), 0);
                            encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                        },
                    )?;
                }
                Qwen38MlpFusion::GateUpPair => {
                    self.encode_fused_gate_up(tcb, layer, false)?;
                    tcb.dispatch_threads(
                        qwen38_swiglu_family_kernel(self.decode_family),
                        (n, 1, 1),
                        (n.min(256).max(1), 1, 1),
                        |encoder| {
                            encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                            encoder.set_buffer(1, Some(&self.workspace.up), 0);
                            encoder.set_buffer(2, Some(&self.workspace.act), 0);
                            encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                        },
                    )?;
                }
                Qwen38MlpFusion::GateUpSwiglu => {
                    self.encode_fused_gate_up(tcb, layer, true)?;
                }
            }
            self.encode_q4_matvec(
                tcb,
                self.layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )?;
            if self.fuse_add_rmsnorm {
                let next = self.next_norm_weight_name(layer);
                self.encode_add_residual_rmsnorm(
                    tcb,
                    input,
                    &self.workspace.down,
                    &self.workspace.hidden,
                    &next,
                    &self.workspace.normalized,
                )
            } else {
                qwen_next_add_residual_tcb(
                    tcb,
                    input,
                    &self.workspace.down,
                    &self.workspace.hidden,
                    QWEN38_HIDDEN,
                )
            }
        }

        fn encode_deltanet(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_deltanet_mixed(tcb, layer);
            }
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = self.deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
            if !(self.fuse_add_rmsnorm && layer > 0) {
                self.encode_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    self.layer_name(layer, "input_layernorm.weight"),
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )?;
            }
            if self.fuse_dn_inproj {
                self.encode_fused_pair_concat(
                    tcb,
                    self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                    &self.workspace.qkvz,
                    self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                    &self.workspace.ba,
                )?;
            } else {
                self.encode_independent_q4_pair(
                    tcb,
                    self.layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
                    &self.workspace.normalized,
                    &self.workspace.qkvz,
                    self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                    &self.workspace.normalized,
                    &self.workspace.ba,
                )?;
            }
            let conv_w = self.f32(self.layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )?;
            self.encode_dn_ba_and_delta(tcb, layer, rec_off)?;
            let norm_w = self.f32(self.layer_name(layer, "linear_attn.norm.weight"))?;
            let dn_tg = self.dn_rmsnorm_tg;
            let (dn_name, dn_grid, dn_tgd) = if dn_tg > 0 {
                ("qwen80_deltanet_gated_rmsnorm_tg",
                 (layout.value_heads as u32 * dn_tg, 1, 1), (dn_tg, 1, 1))
            } else {
                ("qwen80_deltanet_gated_rmsnorm_f32",
                 (layout.value_heads as u32, 1, 1), (16, 1, 1))
            };
            tcb.dispatch_threads(
                dn_name,
                dn_grid,
                dn_tgd,
                |encoder| {
                    if dn_tg > 0 {
                        encoder.set_threadgroup_memory_length(0, (dn_tg as u64) * 4);
                    }
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            self.encode_q4_matvec(
                tcb,
                self.layer_name(layer, "linear_attn.out_proj.weight"),
                &self.workspace.gated,
                &self.workspace.mixer,
            )?;
            if self.fuse_add_rmsnorm {
                self.encode_add_residual_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    self.layer_name(layer, "post_attention_layernorm.weight"),
                    &self.workspace.normalized,
                )
            } else {
                qwen_next_add_residual_tcb(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    QWEN38_HIDDEN,
                )
            }
        }

        fn encode_gqa(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_gqa_mixed(tcb, layer);
            }
            if self.position >= self.max_seq_len {
                return Err(Error::Model(format!(
                    "qwen38 GQA position {} exceeds max_seq_len {}",
                    self.position, self.max_seq_len
                )));
            }
            let slot = self.gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            if !(self.fuse_add_rmsnorm && layer > 0) {
                self.encode_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    self.layer_name(layer, "input_layernorm.weight"),
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )?;
            }
            if self.fuse_gqa_qkv {
                self.encode_fused_qkv(tcb, layer)?;
            } else {
                if self.concurrent_independent {
                    tcb.begin_concurrent_group()?;
                }
                self.encode_q4_matvec(
                    tcb,
                    self.layer_name(layer, "self_attn.q_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.q_proj,
                )?;
                self.encode_q4_matvec(
                    tcb,
                    self.layer_name(layer, "self_attn.k_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.k_proj,
                )?;
                self.encode_q4_matvec(
                    tcb,
                    self.layer_name(layer, "self_attn.v_proj.weight"),
                    &self.workspace.normalized,
                    &self.workspace.v_proj,
                )?;
                if self.concurrent_independent {
                    tcb.end_concurrent_group()?;
                }
            }
            let q_norm = self.f32(self.layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(self.layer_name(layer, "self_attn.k_norm.weight"))?;
            let rope_tg = self.rope_tg;
            let (rope_name, rope_grid, rope_tgd) = if rope_tg > 0 {
                ("qwen38_gqa_qk_norm_rope_cache_tg",
                 (QWEN38_GQA_HEADS as u32 * rope_tg, 1, 1), (rope_tg, 1, 1))
            } else {
                ("qwen38_gqa_qk_norm_rope_cache_f32",
                 (QWEN38_GQA_HEADS as u32, 1, 1), (QWEN38_GQA_HEADS as u32, 1, 1))
            };
            tcb.dispatch_threads(
                rope_name,
                rope_grid,
                rope_tgd,
                |encoder| {
                    if rope_tg > 0 {
                        encoder.set_threadgroup_memory_length(0, (rope_tg as u64) * 4);
                    }
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            self.encode_gqa_mha_and_gate(tcb, cache_off as usize, self.position + 1)?;
            self.encode_q4_matvec(
                tcb,
                self.layer_name(layer, "self_attn.o_proj.weight"),
                &self.workspace.gated_attn,
                &self.workspace.mixer,
            )?;
            if self.fuse_add_rmsnorm {
                self.encode_add_residual_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    self.layer_name(layer, "post_attention_layernorm.weight"),
                    &self.workspace.normalized,
                )
            } else {
                qwen_next_add_residual_tcb(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    QWEN38_HIDDEN,
                )
            }
        }

        fn encode_layers(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            for layer in 0..QWEN38_LAYERS {
                match self.mixer_kind(layer)? {
                    Qwen38MixerKind::DeltaNet => self.encode_deltanet(tcb, layer)?,
                    Qwen38MixerKind::Gqa => self.encode_gqa(tcb, layer)?,
                }
                self.encode_dense_mlp(tcb, layer, &self.workspace.first_residual)?;
            }
            Ok(())
        }

        fn encode_terminal(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            if !self.weights.mixed.is_empty() {
                return self.encode_terminal_mixed(tcb);
            } else {
                if !self.fuse_add_rmsnorm {
                    self.encode_rmsnorm(
                        tcb,
                        &self.workspace.hidden,
                        "language_model.model.norm.weight",
                        &self.workspace.normalized,
                        QWEN38_HIDDEN as u32,
                    )?;
                }
                self.encode_q4_matvec(
                    tcb,
                    "language_model.lm_head.weight",
                    &self.workspace.normalized,
                    &self.workspace.logits,
                )?;
            }
            sample_argmax_f32_tcb(
                tcb,
                &self.workspace.logits,
                &self.workspace.sampled,
                QWEN38_VOCAB,
            )
        }

        fn encode_embed_mixed(&self, tcb: &mut TokenCommandBuffer<'_>, token: u32) -> Result<()> {
            const EMBED: &str = "language_model.model.embed_tokens.weight";
            if let Some(MixedGpuWeight::Affine(weight)) = self.weights.mixed.get(EMBED) {
                if weight.rows != QWEN38_VOCAB as u32 || weight.cols != QWEN38_HIDDEN as u32 {
                    return Err(mixed_error("embed HGRAVF01 shape drifted"));
                }
                let biases = weight.biases.as_ref().ok_or_else(|| {
                    mixed_error("embed HGRAVF01 is delta-only (q2f); embed kernel needs bias")
                })?;
                self.record_affine_embedding_row(weight);
                let hidden = QWEN38_HIDDEN as u32;
                let vocab = QWEN38_VOCAB as u32;
                return tcb.dispatch_threads(
                    QWEN38_HGRAFV_EMBED,
                    (hidden, 1, 1),
                    (256, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&weight.codes), 0);
                        encoder.set_buffer(1, Some(&weight.scales), 0);
                        encoder.set_buffer(2, Some(biases), 0);
                        encoder.set_buffer(3, Some(&self.workspace.hidden), 0);
                        set_u32(encoder, 4, token);
                        set_u32(encoder, 5, hidden);
                        set_u32(encoder, 6, vocab);
                        set_u32(encoder, 7, weight.group_size);
                    },
                );
            }
            if let Some(MixedGpuWeight::Uniform(weight)) = self.weights.mixed.get(EMBED) {
                if weight.rows != QWEN38_VOCAB as u32 || weight.cols != QWEN38_HIDDEN as u32 {
                    return Err(mixed_error("embed HGRAVU01 shape drifted"));
                }
                self.record_uniform_embedding_row(weight);
                let hidden = QWEN38_HIDDEN as u32;
                let vocab = QWEN38_VOCAB as u32;
                return tcb.dispatch_threads(
                    "qwen38_hgravu_embedding_lookup",
                    (hidden, 1, 1),
                    (256, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&weight.codes), 0);
                        encoder.set_buffer(1, Some(&weight.scales), 0);
                        encoder.set_buffer(2, Some(&self.workspace.hidden), 0);
                        set_u32(encoder, 3, token);
                        set_u32(encoder, 4, hidden);
                        set_u32(encoder, 5, vocab);
                        set_u32(encoder, 6, weight.group_size);
                        set_u32(encoder, 7, weight.bits);
                        set_u32(encoder, 8, weight.bound);
                    },
                );
            }
            if self.weights.q4.contains_key(EMBED) {
                let weight = self.q4(EMBED)?;
                if weight.rows != QWEN38_VOCAB || weight.cols != QWEN38_HIDDEN {
                    return Err(Error::Model("qwen38 embed shape drifted".into()));
                }
                self.record_q4_embedding_row(weight);
                let hidden = QWEN38_HIDDEN as u32;
                let vocab = QWEN38_VOCAB as u32;
                let group = weight.group_size as u32;
                return tcb.dispatch_threads(
                    "qwen_uniform_q4_embedding_lookup",
                    (hidden, 1, 1),
                    (256, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&weight.codes), 0);
                        encoder.set_buffer(1, Some(&weight.scales), 0);
                        encoder.set_buffer(2, Some(&self.workspace.hidden), 0);
                        encoder.set_bytes(3, 4, &token as *const u32 as *const _);
                        encoder.set_bytes(4, 4, &hidden as *const u32 as *const _);
                        encoder.set_bytes(5, 4, &vocab as *const u32 as *const _);
                        encoder.set_bytes(6, 4, &group as *const u32 as *const _);
                    },
                );
            }
            Err(mixed_error(
                "embed is neither HGRAVF01 nor HGRAVU01 nor HQ30UQ4; refusing silent fallback",
            ))
        }

        fn encode_dense_mlp_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
            input: &PinnedBuffer,
        ) -> Result<()> {
            let n = QWEN38_INTERMEDIATE as u32;
            if !self.fuse_add_rmsnorm {
                self.encode_rmsnorm(
                    tcb,
                    input,
                    self.layer_name(layer, "post_attention_layernorm.weight"),
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )?;
            }
            match self.mlp_fusion {
                Qwen38MlpFusion::Off => {
                    self.encode_named_matvec(
                        tcb,
                        self.layer_name(layer, "mlp.gate_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.gate,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        self.layer_name(layer, "mlp.up_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.up,
                    )?;
                    tcb.dispatch_threads(
                        qwen38_swiglu_family_kernel(self.decode_family),
                        (n, 1, 1),
                        (n.min(256).max(1), 1, 1),
                        |encoder| {
                            encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                            encoder.set_buffer(1, Some(&self.workspace.up), 0);
                            encoder.set_buffer(2, Some(&self.workspace.act), 0);
                            encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                        },
                    )?;
                }
                Qwen38MlpFusion::GateUpPair => {
                    self.encode_fused_gate_up(tcb, layer, false)?;
                    tcb.dispatch_threads(
                        qwen38_swiglu_family_kernel(self.decode_family),
                        (n, 1, 1),
                        (n.min(256).max(1), 1, 1),
                        |encoder| {
                            encoder.set_buffer(0, Some(&self.workspace.gate), 0);
                            encoder.set_buffer(1, Some(&self.workspace.up), 0);
                            encoder.set_buffer(2, Some(&self.workspace.act), 0);
                            encoder.set_bytes(3, 4, &n as *const u32 as *const _);
                        },
                    )?;
                }
                Qwen38MlpFusion::GateUpSwiglu => {
                    self.encode_fused_gate_up(tcb, layer, true)?;
                }
            }
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )?;
            if self.fuse_add_rmsnorm {
                let next = self.next_norm_weight_name(layer);
                self.encode_add_residual_rmsnorm(
                    tcb,
                    input,
                    &self.workspace.down,
                    &self.workspace.hidden,
                    &next,
                    &self.workspace.normalized,
                )
            } else {
                qwen_next_add_residual_tcb(
                    tcb,
                    input,
                    &self.workspace.down,
                    &self.workspace.hidden,
                    QWEN38_HIDDEN,
                )
            }
        }

        fn encode_deltanet_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            let layout = Qwen38DeltaNetLayout::source_exact();
            let slot = self.deltanet_state_slot(layer)?;
            let conv_off = (slot * layout.conv_state_elements() * 4) as u64;
            let rec_off = (slot * layout.recurrent_state_elements() * 4) as u64;
            if !(self.fuse_add_rmsnorm && layer > 0) {
                self.encode_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    self.layer_name(layer, "input_layernorm.weight"),
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )?;
            }
            let fused_qkvz = self.layer_name(layer, "linear_attn.in_proj_qkvz.weight");
            let fused_ba = self.layer_name(layer, "linear_attn.in_proj_ba.weight");
            if self.has_weight(&fused_qkvz) && self.has_weight(&fused_ba) {
                // The fused pair kernel is the Q4 family. A fast profile may
                // also be used with a heterogeneous catalog, so keep the
                // Q2F/other packed representation on its own native route
                // instead of asking the Q4 binder to reinterpret it.
                if self.fuse_dn_inproj
                    && self.weights.q4.contains_key(fused_qkvz)
                    && self.weights.q4.contains_key(fused_ba)
                {
                    self.encode_fused_pair_concat(
                        tcb,
                        &fused_qkvz,
                        &self.workspace.qkvz,
                        &fused_ba,
                        &self.workspace.ba,
                    )?;
                } else if self.fuse_dn_inproj && self.can_fuse_q2f_pair(layer) {
                    self.encode_fused_q2f_pair(tcb, layer)?;
                } else {
                    self.encode_named_matvec(
                        tcb,
                        &fused_qkvz,
                        &self.workspace.normalized,
                        &self.workspace.qkvz,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        &fused_ba,
                        &self.workspace.normalized,
                        &self.workspace.ba,
                    )?;
                }
            } else if self.has_weight(self.layer_name(layer, "linear_attn.in_proj_qkv.weight")) {
                self.encode_split_deltanet_projections(tcb, layer)?;
            } else {
                return Err(mixed_error(format!(
                    "layer {layer} mixer projections are neither fused QKVZ/BA nor split QKV/Z/A/B"
                )));
            }
            let conv_w = self.f32(self.layer_name(layer, "linear_attn.conv1d.weight"))?;
            tcb.dispatch_threads(
                "qwen38_qkvz_rearrange_conv_l2_f32",
                (256, layout.key_heads as u32, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.workspace.qkvz), 0);
                    encoder.set_buffer(1, Some(conv_w), 0);
                    encoder.set_buffer(2, Some(&self.workspace.conv_state), conv_off);
                    encoder.set_buffer(3, Some(&self.workspace.repeated_q), 0);
                    encoder.set_buffer(4, Some(&self.workspace.repeated_k), 0);
                    encoder.set_buffer(5, Some(&self.workspace.conv_v), 0);
                    encoder.set_buffer(6, Some(&self.workspace.z), 0);
                    let kh = layout.key_heads as u32;
                    let vpk = layout.values_per_key as u32;
                    let kd = layout.key_head_dim as u32;
                    let vd = layout.value_head_dim as u32;
                    let ck = layout.conv_kernel as u32;
                    encoder.set_bytes(7, 4, &kh as *const u32 as *const _);
                    encoder.set_bytes(8, 4, &vpk as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &kd as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &vd as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &ck as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                    encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
                },
            )?;
            self.encode_dn_ba_and_delta(tcb, layer, rec_off)?;
            let norm_w = self.f32(self.layer_name(layer, "linear_attn.norm.weight"))?;
            let dn_tg = self.dn_rmsnorm_tg;
            let (dn_name, dn_grid, dn_tgd) = if dn_tg > 0 {
                ("qwen80_deltanet_gated_rmsnorm_tg",
                 (layout.value_heads as u32 * dn_tg, 1, 1), (dn_tg, 1, 1))
            } else {
                ("qwen80_deltanet_gated_rmsnorm_f32",
                 (layout.value_heads as u32, 1, 1), (16, 1, 1))
            };
            tcb.dispatch_threads(
                dn_name,
                dn_grid,
                dn_tgd,
                |encoder| {
                    if dn_tg > 0 {
                        encoder.set_threadgroup_memory_length(0, (dn_tg as u64) * 4);
                    }
                    encoder.set_buffer(0, Some(&self.workspace.rec_out), 0);
                    encoder.set_buffer(1, Some(&self.workspace.z), 0);
                    encoder.set_buffer(2, Some(norm_w), 0);
                    encoder.set_buffer(3, Some(&self.workspace.gated), 0);
                    let heads = layout.value_heads as u32;
                    let dim = layout.value_head_dim as u32;
                    encoder.set_bytes(4, 4, &heads as *const u32 as *const _);
                    encoder.set_bytes(5, 4, &dim as *const u32 as *const _);
                    encoder.set_bytes(6, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "linear_attn.out_proj.weight"),
                &self.workspace.gated,
                &self.workspace.mixer,
            )?;
            if self.fuse_add_rmsnorm {
                self.encode_add_residual_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    self.layer_name(layer, "post_attention_layernorm.weight"),
                    &self.workspace.normalized,
                )
            } else {
                qwen_next_add_residual_tcb(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    QWEN38_HIDDEN,
                )
            }
        }

        fn encode_gqa_mixed(&self, tcb: &mut TokenCommandBuffer<'_>, layer: usize) -> Result<()> {
            if self.position >= self.max_seq_len {
                return Err(Error::Model(format!(
                    "qwen38 GQA position {} exceeds max_seq_len {}",
                    self.position, self.max_seq_len
                )));
            }
            let slot = self.gqa_state_slot(layer)?;
            let slot_elems = self.max_seq_len * QWEN38_GQA_KV_HEADS * QWEN38_GQA_HEAD_DIM;
            let cache_off = (slot * slot_elems * 4) as u64;
            if !(self.fuse_add_rmsnorm && layer > 0) {
                self.encode_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    self.layer_name(layer, "input_layernorm.weight"),
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )?;
            }
            let q_name = self.layer_name(layer, "self_attn.q_proj.weight");
            let k_name = self.layer_name(layer, "self_attn.k_proj.weight");
            let v_name = self.layer_name(layer, "self_attn.v_proj.weight");
            if self.fuse_gqa_qkv
                && self.weights.q4.contains_key(q_name)
                && self.weights.q4.contains_key(k_name)
                && self.weights.q4.contains_key(v_name)
            {
                self.encode_fused_qkv(tcb, layer)?;
            } else if self.fuse_gqa_qkv && self.can_fuse_q2f_qkv(layer) {
                self.encode_fused_q2f_qkv(tcb, layer)?;
            } else {
                self.encode_named_matvec(
                    tcb,
                    &q_name,
                    &self.workspace.normalized,
                    &self.workspace.q_proj,
                )?;
                self.encode_named_matvec(
                    tcb,
                    &k_name,
                    &self.workspace.normalized,
                    &self.workspace.k_proj,
                )?;
                self.encode_named_matvec(
                    tcb,
                    &v_name,
                    &self.workspace.normalized,
                    &self.workspace.v_proj,
                )?;
            }
            let q_norm = self.f32(self.layer_name(layer, "self_attn.q_norm.weight"))?;
            let k_norm = self.f32(self.layer_name(layer, "self_attn.k_norm.weight"))?;
            let rope_tg = self.rope_tg;
            let (rope_name, rope_grid, rope_tgd) = if rope_tg > 0 {
                ("qwen38_gqa_qk_norm_rope_cache_tg",
                 (QWEN38_GQA_HEADS as u32 * rope_tg, 1, 1), (rope_tg, 1, 1))
            } else {
                ("qwen38_gqa_qk_norm_rope_cache_f32",
                 (QWEN38_GQA_HEADS as u32, 1, 1), (QWEN38_GQA_HEADS as u32, 1, 1))
            };
            tcb.dispatch_threads(
                rope_name,
                rope_grid,
                rope_tgd,
                |encoder| {
                    if rope_tg > 0 {
                        encoder.set_threadgroup_memory_length(0, (rope_tg as u64) * 4);
                    }
                    encoder.set_buffer(0, Some(&self.workspace.q_proj), 0);
                    encoder.set_buffer(1, Some(&self.workspace.k_proj), 0);
                    encoder.set_buffer(2, Some(&self.workspace.v_proj), 0);
                    encoder.set_buffer(3, Some(q_norm), 0);
                    encoder.set_buffer(4, Some(k_norm), 0);
                    encoder.set_buffer(5, Some(&self.workspace.query), 0);
                    encoder.set_buffer(6, Some(&self.workspace.gqa_key), cache_off);
                    encoder.set_buffer(7, Some(&self.workspace.gqa_value), cache_off);
                    let pos = self.position as u32;
                    let nh = QWEN38_GQA_HEADS as u32;
                    let nkv = QWEN38_GQA_KV_HEADS as u32;
                    let hd = QWEN38_GQA_HEAD_DIM as u32;
                    let rd = QWEN38_GQA_ROTARY_DIM as u32;
                    encoder.set_bytes(8, 4, &pos as *const u32 as *const _);
                    encoder.set_bytes(9, 4, &nh as *const u32 as *const _);
                    encoder.set_bytes(10, 4, &nkv as *const u32 as *const _);
                    encoder.set_bytes(11, 4, &hd as *const u32 as *const _);
                    encoder.set_bytes(12, 4, &rd as *const u32 as *const _);
                    encoder.set_bytes(13, 4, &QWEN38_ROPE_THETA as *const f32 as *const _);
                    encoder.set_bytes(14, 4, &QWEN38_RMS_EPS as *const f32 as *const _);
                },
            )?;
            self.encode_gqa_mha_and_gate(tcb, cache_off as usize, self.position + 1)?;
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "self_attn.o_proj.weight"),
                &self.workspace.gated_attn,
                &self.workspace.mixer,
            )?;
            if self.fuse_add_rmsnorm {
                self.encode_add_residual_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    self.layer_name(layer, "post_attention_layernorm.weight"),
                    &self.workspace.normalized,
                )
            } else {
                qwen_next_add_residual_tcb(
                    tcb,
                    &self.workspace.hidden,
                    &self.workspace.mixer,
                    &self.workspace.first_residual,
                    QWEN38_HIDDEN,
                )
            }
        }

        fn encode_terminal_mixed(&self, tcb: &mut TokenCommandBuffer<'_>) -> Result<()> {
            if !self.fuse_add_rmsnorm {
                self.encode_rmsnorm(
                    tcb,
                    &self.workspace.hidden,
                    "language_model.model.norm.weight",
                    &self.workspace.normalized,
                    QWEN38_HIDDEN as u32,
                )?;
            }
            self.encode_named_matvec(
                tcb,
                "language_model.lm_head.weight",
                &self.workspace.normalized,
                &self.workspace.logits,
            )?;
            sample_argmax_f32_tcb(
                tcb,
                &self.workspace.logits,
                &self.workspace.sampled,
                QWEN38_VOCAB,
            )
        }

        fn encode_mixer_gemvs_only_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            match self.mixer_kind(layer)? {
                Qwen38MixerKind::DeltaNet => {
                    let fused = self.layer_name(layer, "linear_attn.in_proj_qkvz.weight");
                    if self.has_weight(&fused) {
                        if self.fuse_dn_inproj && self.can_fuse_q2f_pair(layer) {
                            self.encode_fused_q2f_pair(tcb, layer)?;
                        } else {
                            self.encode_named_matvec(
                                tcb,
                                &fused,
                                &self.workspace.normalized,
                                &self.workspace.qkvz,
                            )?;
                            self.encode_named_matvec(
                                tcb,
                                self.layer_name(layer, "linear_attn.in_proj_ba.weight"),
                                &self.workspace.normalized,
                                &self.workspace.ba,
                            )?;
                        }
                    } else {
                        self.encode_split_deltanet_projections(tcb, layer)?;
                    }
                    self.encode_named_matvec(
                        tcb,
                        self.layer_name(layer, "linear_attn.out_proj.weight"),
                        &self.workspace.gated,
                        &self.workspace.mixer,
                    )
                }
                Qwen38MixerKind::Gqa => {
                    self.encode_named_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.q_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.q_proj,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.k_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.k_proj,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.v_proj.weight"),
                        &self.workspace.normalized,
                        &self.workspace.v_proj,
                    )?;
                    self.encode_named_matvec(
                        tcb,
                        self.layer_name(layer, "self_attn.o_proj.weight"),
                        &self.workspace.gated_attn,
                        &self.workspace.mixer,
                    )
                }
            }
        }

        fn encode_mlp_matvecs_only_mixed(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            layer: usize,
        ) -> Result<()> {
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "mlp.gate_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.gate,
            )?;
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "mlp.up_proj.weight"),
                &self.workspace.normalized,
                &self.workspace.up,
            )?;
            self.encode_named_matvec(
                tcb,
                self.layer_name(layer, "mlp.down_proj.weight"),
                &self.workspace.act,
                &self.workspace.down,
            )
        }

        pub fn step(&mut self, token: u32) -> Result<(u32, CommandBufferTiming)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            self.reset_active_weight_bytes();
            let encode_t0 = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.enable_dispatch_name_trace(&mut tcb);
            let encode_result = self.encode_full_token(&mut tcb, token);
            encode_result?;
            let harvested = tcb.structural_kernel_names().map(|names| names.to_vec());
            let encode_ns = encode_t0.elapsed().as_nanos() as u64;
            let mut timing = tcb.commit_and_wait_timed()?;
            self.harvest_dispatch_names(harvested);
            if timing.encode_ns == 0 {
                timing.encode_ns = encode_ns;
            }
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            self.position = self.position.saturating_add(1);
            Ok((sampled, timing))
        }

        /// Same device work and state transition as [`Self::step`], but for a
        /// resident serving path that does not request per-token timing.  The
        /// plain fence path avoids host clock reads and GPU timestamp queries
        /// when the attached context is untraced and the cost ledger is off.
        ///
        /// This is intentionally separate from `step`: benchmark and
        /// qualification callers keep their complete timing contract, while a
        /// serving caller can opt into the lower-ceremony path explicitly.
        pub fn step_unmeasured(&mut self, token: u32) -> Result<u32> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            self.reset_active_weight_bytes();
            let previous_active_weight_accounting = self.active_weight_accounting;
            self.active_weight_accounting = false;
            let mut tcb = TokenCommandBuffer::new(&self.context);
            self.enable_dispatch_name_trace(&mut tcb);
            let encode_result = self.encode_full_token(&mut tcb, token);
            self.active_weight_accounting = previous_active_weight_accounting;
            encode_result?;
            let harvested = tcb.structural_kernel_names().map(|names| names.to_vec());
            tcb.commit_and_wait()?;
            self.harvest_dispatch_names(harvested);
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            self.position = self.position.saturating_add(1);
            Ok(sampled)
        }

        /// Same GPU work as [`Self::step`], with host-side Instants around
        /// encode / commit-return / sample readback / position update so
        /// wall − gpu can be named. Timers do not change the command buffer.
        pub fn step_complete(&mut self, token: u32) -> Result<(u32, Qwen38StepWall)> {
            if self.fallbacks != 0 {
                return Err(Error::Model(
                    "qwen38 decode refuses a run after a fallback".into(),
                ));
            }
            self.reset_active_weight_bytes();
            let wall = Instant::now();
            let alloc_started = Instant::now();
            let mut tcb = TokenCommandBuffer::new(&self.context);
            let allocation_ns = alloc_started.elapsed().as_nanos() as u64;
            self.enable_dispatch_name_trace(&mut tcb);
            let encode_started = Instant::now();
            let encode_result = self.encode_full_token(&mut tcb, token);
            encode_result?;
            let harvested = tcb.structural_kernel_names().map(|names| names.to_vec());
            let encode_ns = encode_started.elapsed().as_nanos() as u64;
            let commit_started = Instant::now();
            let timing = tcb.commit_and_wait_timed()?;
            self.harvest_dispatch_names(harvested);
            let commit_return_ns = commit_started.elapsed().as_nanos() as u64;
            let submit_plus_wait = timing.submit_ns.saturating_add(timing.wait_ns);
            let commit_epilogue_ns = commit_return_ns.saturating_sub(submit_plus_wait);
            let readback_started = Instant::now();
            let sampled = unsafe { *(self.workspace.sampled.contents() as *const u32) };
            let sample_readback_ns = readback_started.elapsed().as_nanos() as u64;
            let state_started = Instant::now();
            self.position = self.position.saturating_add(1);
            let state_update_ns = state_started.elapsed().as_nanos() as u64;
            let command_buffers = if timing.command_buffers == 0 {
                1
            } else {
                timing.command_buffers
            };
            Ok((
                sampled,
                Qwen38StepWall {
                    wall_ns: wall.elapsed().as_nanos() as u64,
                    encode_ns,
                    submit_ns: timing.submit_ns,
                    wait_ns: timing.wait_ns,
                    gpu_ns: timing.gpu_ns,
                    gpu_start_s: timing.gpu_start_s,
                    gpu_end_s: timing.gpu_end_s,
                    gpu_start_ns: timing.gpu_start_ns,
                    gpu_end_ns: timing.gpu_end_ns,
                    allocation_ns,
                    encoder_count: timing.encoder_count,
                    commit_epilogue_ns,
                    sample_readback_ns,
                    state_update_ns,
                    tcb_encode_ns: timing.encode_ns,
                    dispatches: timing.dispatches,
                    command_buffers,
                    active_weight_bytes: self.last_active_weight_bytes(),
                },
            ))
        }
    }

    pub fn generate_greedy(
        session: &mut Qwen38HybridDecodeSession,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Qwen38GenerateResult> {
        generate_greedy_reusing(session, prompt, max_new_tokens, 0)
    }

    /// `generate_greedy`, but the first `reuse` prompt tokens are already
    /// resident in the session's KV and recurrent state and are NOT re-stepped.
    ///
    /// `reuse == 0` resets and prefills everything -- byte-identical to the old
    /// behaviour, which is what `generate_greedy` still is.
    ///
    /// PURE APPEND ONLY. The caller must have verified that the resident
    /// context is an exact prefix of `prompt`. DeltaNet state is RECURRENT: it
    /// is a running summary with no per-position index, so it cannot be
    /// truncated or rewound. Reusing a prefix that diverged would silently
    /// condition generation on tokens that are not in the prompt.
    pub fn generate_greedy_reusing(
        session: &mut Qwen38HybridDecodeSession,
        prompt: &[u32],
        max_new_tokens: usize,
        reuse: usize,
    ) -> Result<Qwen38GenerateResult> {
        generate_greedy_reusing_snapshot(session, prompt, max_new_tokens, reuse, None)
            .map(|(result, _)| result)
    }

    /// As `generate_greedy_reusing`, and additionally captures a prefix
    /// checkpoint the moment `position` reaches `snapshot_at`.
    ///
    /// Taken DURING work already being done: the caller cannot snapshot a
    /// boundary after prefilling past it, because the recurrent carry has
    /// already moved on and there is no way to rewind it.
    pub fn generate_greedy_reusing_snapshot(
        session: &mut Qwen38HybridDecodeSession,
        prompt: &[u32],
        max_new_tokens: usize,
        reuse: usize,
        snapshot_at: Option<usize>,
    ) -> Result<(Qwen38GenerateResult, Option<Qwen38PrefixCheckpoint>)> {
        let result = generate_greedy_reusing_inner(
            session,
            prompt,
            max_new_tokens,
            reuse,
            snapshot_at,
        )?;
        Ok(result)
    }

    fn generate_greedy_reusing_inner(
        session: &mut Qwen38HybridDecodeSession,
        prompt: &[u32],
        max_new_tokens: usize,
        reuse: usize,
        snapshot_at: Option<usize>,
    ) -> Result<(Qwen38GenerateResult, Option<Qwen38PrefixCheckpoint>)> {
        let mut snapshot: Option<Qwen38PrefixCheckpoint> = None;
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        if reuse >= prompt.len() {
            // At least one token must be stepped: `next` is the argmax the last
            // stepped token produced, and with nothing stepped there is none.
            return Err(Error::Model(format!(
                "qwen38 reuse {reuse} leaves no prompt token to step (prompt is {})",
                prompt.len()
            )));
        }
        if reuse == 0 {
            session.reset();
        }
        // Every prompt token is stepped once, followed by at most
        // `max_new_tokens` sampled ids. Reserve the complete known envelope so
        // request-result growth never adds realloc/copy work to the measured
        // host path.
        let step_capacity = prompt.len().saturating_add(max_new_tokens);
        let token_capacity = step_capacity.saturating_add(1);
        let mut tokens = Vec::with_capacity(token_capacity);
        tokens.extend_from_slice(prompt);
        let mut gpu_ns = Vec::with_capacity(step_capacity);
        let mut wait_ns = Vec::with_capacity(step_capacity);
        let mut encode_ns = Vec::with_capacity(step_capacity);
        let mut submit_ns = Vec::with_capacity(step_capacity);
        let mut dispatches = Vec::with_capacity(step_capacity);
        let mut active_weight_bytes = Vec::with_capacity(step_capacity);
        let mut wall_ns_per_step = Vec::with_capacity(step_capacity);
        let wall = Instant::now();
        let mut next = 0u32;
        let prefill = Instant::now();
        let mut first_step_wall_ns = 0u64;
        for (i, &token) in prompt.iter().enumerate().skip(reuse) {
            let step_wall = Instant::now();
            let (sampled, timing) = session.step(token)?;
            let step_ns = step_wall.elapsed().as_nanos() as u64;
            if i == reuse {
                first_step_wall_ns = step_ns;
            }
            wall_ns_per_step.push(step_ns);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            active_weight_bytes.push(session.last_active_weight_bytes());
            next = sampled;
            if snapshot.is_none() && snapshot_at == Some(i + 1) {
                // i + 1 prompt tokens have been stepped, so the carry is exactly
                // the state after that prefix. Captured HERE because it cannot
                // be recovered once the prefill moves past it: the recurrent
                // state is a running summary with no rewind.
                snapshot = Some(session.prefix_checkpoint()?);
            }
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        tokens.push(next);
        let decode = Instant::now();
        let ignore_eos = std::env::var("HAWKING_QWEN38_IGNORE_EOS")
            .map(|v| v != "0")
            .unwrap_or(false);
        while tokens.len() - prompt.len() < max_new_tokens {
            if !ignore_eos
                && (next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                    || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT)
            {
                break;
            }
            let step_wall = Instant::now();
            let (sampled, timing) = session.step(next)?;
            wall_ns_per_step.push(step_wall.elapsed().as_nanos() as u64);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            active_weight_bytes.push(session.last_active_weight_bytes());
            tokens.push(sampled);
            next = sampled;
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        let decode_steps = tokens.len().saturating_sub(prompt.len()).saturating_sub(1);
        Ok((Qwen38GenerateResult {
            stop_reason: "",
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            gpu_ns,
            wait_ns,
            encode_ns,
            submit_ns,
            dispatches,
            active_weight_bytes,
            fallbacks: session.fallbacks,
            dense_w_materialized: session.dense_w_materialized,
            resident_weight_bytes: session.resident_weight_bytes(),
            workspace_resident_bytes: session.workspace_resident_bytes(),
            first_step_wall_ns,
            prefill_wall_ns,
            decode_wall_ns,
            decode_steps,
            wall_ns_per_step,
        }, snapshot))
    }

    /// Greedy generation with a JSON logit mask applied on the host.
    ///
    /// The GPU argmax kernel still runs inside [`Qwen38HybridDecodeSession::step`];
    /// its answer is discarded. After each wait, logits are read from the
    /// shared workspace, masked, and reduced with the Metal tie-break
    /// (strictly greater wins, exact ties keep the lower index).
    pub fn generate_constrained(
        session: &mut Qwen38HybridDecodeSession,
        tokenizer: &Tokenizer,
        vocab: &JsonVocabIndex,
        constraint: &mut JsonConstraint,
        prompt: &[u32],
        max_new_tokens: usize,
        reuse: usize,
        snapshot_at: Option<usize>,
    ) -> Result<(Qwen38GenerateResult, Option<Qwen38PrefixCheckpoint>)> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        if reuse >= prompt.len() {
            return Err(Error::Model(format!(
                "qwen38 reuse {reuse} leaves no prompt token to step (prompt is {})",
                prompt.len()
            )));
        }
        let mut snapshot: Option<Qwen38PrefixCheckpoint> = None;
        // The constrained path resets ONLY on a cold request now. It used to
        // reset unconditionally, which made the grammar channel and the prefix
        // cache mutually exclusive -- and since the grammar channel is on, the
        // cache could never fire in the shipped configuration.
        if reuse == 0 {
            session.reset();
        }
        let step_capacity = prompt.len().saturating_add(max_new_tokens);
        let token_capacity = step_capacity.saturating_add(1);
        let mut tokens = Vec::with_capacity(token_capacity);
        tokens.extend_from_slice(prompt);
        let mut gpu_ns = Vec::with_capacity(step_capacity);
        let mut wait_ns = Vec::with_capacity(step_capacity);
        let mut encode_ns = Vec::with_capacity(step_capacity);
        let mut submit_ns = Vec::with_capacity(step_capacity);
        let mut dispatches = Vec::with_capacity(step_capacity);
        let mut active_weight_bytes = Vec::with_capacity(step_capacity);
        let mut wall_ns_per_step = Vec::with_capacity(step_capacity);
        let wall = Instant::now();
        let prefill = Instant::now();
        let mut first_step_wall_ns = 0u64;
        for (i, &token) in prompt.iter().enumerate().skip(reuse) {
            let step_wall = Instant::now();
            let (_, timing) = session.step(token)?;
            let step_ns = step_wall.elapsed().as_nanos() as u64;
            if i == reuse {
                first_step_wall_ns = step_ns;
            }
            wall_ns_per_step.push(step_ns);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            active_weight_bytes.push(session.last_active_weight_bytes());
            if snapshot.is_none() && snapshot_at == Some(i + 1) {
                snapshot = Some(session.prefix_checkpoint()?);
            }
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        // Last prefill step already dispatched GPU argmax; discard it and pick
        // the first generated id on the host so the JSON mask applies.
        let mut logits = session.read_f32_workspace("logits", QWEN38_VOCAB)?;
        constraint.mask_logits(vocab, &mut logits);
        // A state where the mask leaves nothing legal must SAY so. Without this
        // the argmax over an all-NEG_INF vector returns id 0 and the resident
        // emits token 0 to the budget while still reporting grammar_enforced --
        // a silent wrong answer wearing an enforcement claim.
        if !logits.iter().any(|v| v.is_finite()) {
            return Err(Error::Model(
                "json constraint masked every token at the first generated position".into(),
            ));
        }
        let mut next = argmax_f32_metal_tiebreak(&logits);
        tokens.push(next);
        constraint.advance(&tokenizer.decode_one(next).unwrap_or_default());
        let decode = Instant::now();
        let ignore_eos = std::env::var("HAWKING_QWEN38_IGNORE_EOS")
            .map(|v| v != "0")
            .unwrap_or(false);
        // WHY generation stopped. "never closed the JSON object" has three
        // possible causes -- the constraint believed it closed, the model emitted
        // EOS, or the budget ran out -- and they need different fixes. Without
        // this the receipt reports the symptom and the cause has to be guessed.
        let mut stop_reason = "budget";
        while tokens.len() - prompt.len() < max_new_tokens {
            if constraint.is_done() {
                stop_reason = "constraint_done";
                break;
            }
            if !ignore_eos
                && (next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                    || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT)
            {
                stop_reason = "eos";
                break;
            }
            let step_wall = Instant::now();
            let (_, timing) = session.step(next)?;
            wall_ns_per_step.push(step_wall.elapsed().as_nanos() as u64);
            gpu_ns.push(timing.gpu_ns);
            wait_ns.push(timing.wait_ns);
            encode_ns.push(timing.encode_ns);
            submit_ns.push(timing.submit_ns);
            dispatches.push(timing.dispatches);
            active_weight_bytes.push(session.last_active_weight_bytes());
            let mut logits = session.read_f32_workspace("logits", QWEN38_VOCAB)?;
            constraint.mask_logits(vocab, &mut logits);
            if !logits.iter().any(|v| v.is_finite()) {
                return Err(Error::Model(format!(
                    "json constraint masked every token at generated position {}",
                    tokens.len() - prompt.len()
                )));
            }
            let sampled = argmax_f32_metal_tiebreak(&logits);
            tokens.push(sampled);
            next = sampled;
            constraint.advance(&tokenizer.decode_one(sampled).unwrap_or_default());
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        let decode_steps = tokens.len().saturating_sub(prompt.len()).saturating_sub(1);
        Ok((Qwen38GenerateResult {
            stop_reason: "",
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            gpu_ns,
            wait_ns,
            encode_ns,
            submit_ns,
            dispatches,
            active_weight_bytes,
            fallbacks: session.fallbacks,
            dense_w_materialized: session.dense_w_materialized,
            resident_weight_bytes: session.resident_weight_bytes(),
            workspace_resident_bytes: session.workspace_resident_bytes(),
            first_step_wall_ns,
            prefill_wall_ns,
            decode_wall_ns,
            decode_steps,
            wall_ns_per_step,
        }, snapshot))
    }

    /// Greedy generation for an explicitly selected resident serving path.
    ///
    /// Unlike [`generate_greedy`], this does not allocate per-token timing
    /// vectors or call host clocks around every token.  The returned
    /// `Qwen38GenerateResult` intentionally leaves those vectors empty; its
    /// aggregate phase clocks are retained for coarse RPC observability, and
    /// callers must not interpret the empty vectors as zero GPU work.
    pub fn generate_greedy_unmeasured(
        session: &mut Qwen38HybridDecodeSession,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Qwen38GenerateResult> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        session.reset();
        let capacity = prompt.len().saturating_add(max_new_tokens).saturating_add(1);
        let mut tokens = Vec::with_capacity(capacity);
        tokens.extend_from_slice(prompt);
        let wall = Instant::now();
        let prefill = Instant::now();
        let mut next = 0u32;
        for &token in prompt {
            next = session.step_unmeasured(token)?;
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        tokens.push(next);
        let decode = Instant::now();
        let ignore_eos = std::env::var("HAWKING_QWEN38_IGNORE_EOS")
            .map(|v| v != "0")
            .unwrap_or(false);
        while tokens.len() - prompt.len() < max_new_tokens {
            if !ignore_eos
                && (next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                    || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT)
            {
                break;
            }
            next = session.step_unmeasured(next)?;
            tokens.push(next);
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        let decode_steps = tokens.len().saturating_sub(prompt.len()).saturating_sub(1);
        Ok(Qwen38GenerateResult {
            stop_reason: "",
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            gpu_ns: Vec::new(),
            wait_ns: Vec::new(),
            encode_ns: Vec::new(),
            submit_ns: Vec::new(),
            dispatches: Vec::new(),
            active_weight_bytes: Vec::new(),
            fallbacks: session.fallbacks,
            dense_w_materialized: session.dense_w_materialized,
            resident_weight_bytes: session.resident_weight_bytes(),
            workspace_resident_bytes: session.workspace_resident_bytes(),
            first_step_wall_ns: 0,
            prefill_wall_ns,
            decode_wall_ns,
            decode_steps,
            wall_ns_per_step: Vec::new(),
        })
    }

    pub fn generate_greedy_complete_wall(
        session: &mut Qwen38HybridDecodeSession,
        tokenizer: &Tokenizer,
        prompt: &[u32],
        max_new_tokens: usize,
    ) -> Result<Qwen38CompleteWallResult> {
        if prompt.is_empty() {
            return Err(Error::Model("qwen38 prompt is empty".into()));
        }
        let reset_started = Instant::now();
        session.reset();
        let reset_ns = reset_started.elapsed().as_nanos() as u64;
        // The complete-token profile is itself the latency authority.  The
        // request envelope is known up front, so reserve it before the first
        // measured step and keep Vec growth out of the token loop.
        let step_capacity = prompt.len().saturating_add(max_new_tokens);
        let mut tokens = Vec::with_capacity(step_capacity.saturating_add(1));
        tokens.extend_from_slice(prompt);
        let mut steps = Vec::with_capacity(step_capacity);
        let wall = Instant::now();
        let mut next = 0u32;
        let prefill = Instant::now();
        for (i, &token) in prompt.iter().enumerate() {
            let complete = Instant::now();
            let (sampled, step) = session.step_complete(token)?;
            let last_prompt = i + 1 == prompt.len();
            let (tokenizer_decode_ns, bookkeeping_ns) = if last_prompt {
                finish_new_token(tokenizer, &mut tokens, sampled)?
            } else {
                next = sampled;
                (0, 0)
            };
            if last_prompt {
                next = sampled;
            }
            steps.push(Qwen38CompleteToken {
                role: if last_prompt {
                    "prefill_emits_first_new"
                } else {
                    "prefill"
                },
                step_index: i,
                token_in: token,
                token_out: sampled,
                step,
                tokenizer_decode_ns,
                bookkeeping_ns,
                complete_wall_ns: complete.elapsed().as_nanos() as u64,
            });
        }
        let prefill_wall_ns = prefill.elapsed().as_nanos() as u64;
        let decode = Instant::now();
        while tokens.len() - prompt.len() < max_new_tokens {
            if next == crate::model::qwen38_geometry::QWEN38_EOS_IM_END
                || next == crate::model::qwen38_geometry::QWEN38_EOS_END_OF_TEXT
            {
                break;
            }
            let complete = Instant::now();
            let (sampled, step) = session.step_complete(next)?;
            let (tokenizer_decode_ns, bookkeeping_ns) =
                finish_new_token(tokenizer, &mut tokens, sampled)?;
            steps.push(Qwen38CompleteToken {
                role: "decode",
                step_index: steps.len(),
                token_in: next,
                token_out: sampled,
                step,
                tokenizer_decode_ns,
                bookkeeping_ns,
                complete_wall_ns: complete.elapsed().as_nanos() as u64,
            });
            next = sampled;
        }
        let decode_wall_ns = decode.elapsed().as_nanos() as u64;
        Ok(Qwen38CompleteWallResult {
            tokens,
            prompt_len: prompt.len(),
            wall_ns: wall.elapsed().as_nanos() as u64,
            reset_ns,
            prefill_wall_ns,
            decode_wall_ns,
            fallbacks: session.fallbacks,
            dense_w_materialized: session.dense_w_materialized,
            resident_weight_bytes: session.resident_weight_bytes(),
            workspace_resident_bytes: session.workspace_resident_bytes(),
            steps,
        })
    }

    fn finish_new_token(
        tokenizer: &Tokenizer,
        tokens: &mut Vec<u32>,
        sampled: u32,
    ) -> Result<(u64, u64)> {
        let tokenizer_started = Instant::now();
        tokenizer.decode(&[sampled], true)?;
        let tokenizer_decode_ns = tokenizer_started.elapsed().as_nanos() as u64;
        let bookkeeping_started = Instant::now();
        tokens.push(sampled);
        let bookkeeping_ns = bookkeeping_started.elapsed().as_nanos() as u64;
        Ok((tokenizer_decode_ns, bookkeeping_ns))
    }

    #[derive(Clone, Debug, serde::Serialize)]
    pub struct Qwen38WeightFanout {
        pub weight_name: String,
        pub sessions: usize,
        pub concurrent: bool,
        pub gpu_ns: Option<u64>,
        pub wait_ns: u64,
        pub dispatches: u64,
    }

    /// N independent GEMVs against one resident weight tensor.
    /// `concurrent` opens one Metal concurrent encoder so the GPU may reuse
    /// the weight stream; serial is N dispatches in the default encoder.
    pub fn measure_shared_weight_fanout(
        sessions: &[&Qwen38HybridDecodeSession],
        weight_name: &str,
        concurrent: bool,
    ) -> Result<Qwen38WeightFanout> {
        if sessions.is_empty() {
            return Err(Error::Model("fanout needs at least one session".into()));
        }
        for session in sessions.iter().skip(1) {
            if !sessions[0].shares_weights_with(session) {
                return Err(Error::Model(
                    "fanout sessions do not share one resident weight set".into(),
                ));
            }
        }
        let mut tcb = TokenCommandBuffer::new(&sessions[0].context);
        if concurrent && sessions.len() > 1 {
            tcb.begin_concurrent_group()?;
            if let Ok(weight) = sessions[0].q4(weight_name) {
                tcb.use_resources_read_on_group(&[weight.codes.clone(), weight.scales.clone()])?;
            }
        }
        for session in sessions {
            session.encode_named_matvec(
                &mut tcb,
                weight_name,
                &session.workspace.normalized,
                &session.workspace.logits,
            )?;
        }
        if concurrent && sessions.len() > 1 {
            tcb.end_concurrent_group()?;
        }
        let timing = tcb.commit_and_wait_timed()?;
        Ok(Qwen38WeightFanout {
            weight_name: weight_name.to_owned(),
            sessions: sessions.len(),
            concurrent,
            gpu_ns: timing.gpu_ns,
            wait_ns: timing.wait_ns,
            dispatches: timing.dispatches,
        })
    }

    pub fn generate_greedy_parallel(
        sessions: &mut [Qwen38HybridDecodeSession],
        prompts: &[Vec<u32>],
        max_new_tokens: usize,
    ) -> Result<Vec<Qwen38GenerateResult>> {
        if sessions.len() != prompts.len() {
            return Err(Error::Model(
                "generate_greedy_parallel session/prompt count mismatch".into(),
            ));
        }
        if sessions.len() <= 1 {
            return sessions
                .iter_mut()
                .zip(prompts.iter())
                .map(|(session, prompt)| generate_greedy(session, prompt, max_new_tokens))
                .collect();
        }
        thread::scope(|scope| {
            let mut joins = Vec::with_capacity(sessions.len());
            for (session, prompt) in sessions.iter_mut().zip(prompts.iter()) {
                joins.push(scope.spawn(move || generate_greedy(session, prompt, max_new_tokens)));
            }
            joins
                .into_iter()
                .map(|join| {
                    join.join().unwrap_or_else(|_| {
                        Err(Error::Model("session thread panicked".into()))
                    })
                })
                .collect()
        })
    }

    #[cfg(test)]
    mod active_weight_accounting_tests {
        use super::Qwen38HybridDecodeSession;

        #[test]
        fn embedding_row_accounting_divides_each_packed_plane() {
            assert_eq!(Qwen38HybridDecodeSession::bytes_per_row(4, &[8, 4]), 3);
            assert_eq!(Qwen38HybridDecodeSession::bytes_per_row(0, &[8, 4]), 0);
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Qwen38StepWall {
    pub wall_ns: u64,
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: Option<u64>,
    /// Absolute `GPUStartTime` seconds on the driver epoch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_start_s: Option<f64>,
    /// Absolute `GPUEndTime` seconds, paired with `gpu_start_s`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_end_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_start_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_end_ns: Option<u64>,
    /// Host Instant around `TokenCommandBuffer::new`. Split out of encode so
    /// allocation is a classified idle cause, not mixed into command construction.
    pub allocation_ns: u64,
    pub encoder_count: u64,
    /// `commit_and_wait_timed` return minus submit minus wait: GPU timestamp
    /// read + command-buffer status check after the host wait returns.
    pub commit_epilogue_ns: u64,
    pub sample_readback_ns: u64,
    pub state_update_ns: u64,
    /// TCB per-dispatch encode sum. Zero unless the cost ledger is recording.
    pub tcb_encode_ns: u64,
    pub dispatches: u64,
    pub command_buffers: u64,
    /// Packed weight payload bytes accounted at encoder bind sites for this
    /// token. It excludes activation/state traffic and hardware cache reads.
    pub active_weight_bytes: u64,
}

impl Qwen38StepWall {
    pub fn named_sum_ns(&self) -> u64 {
        self.allocation_ns
            .saturating_add(self.encode_ns)
            .saturating_add(self.submit_ns)
            .saturating_add(self.wait_ns)
            .saturating_add(self.commit_epilogue_ns)
            .saturating_add(self.sample_readback_ns)
            .saturating_add(self.state_update_ns)
    }

    pub fn residual_ns(&self) -> i64 {
        self.wall_ns as i64 - self.named_sum_ns() as i64
    }

    pub fn wait_minus_gpu_ns(&self) -> Option<i64> {
        Some(self.wait_ns as i64 - self.gpu_ns? as i64)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen38CompleteToken {
    /// Fixed protocol labels; borrowing static storage avoids one heap
    /// allocation per profiled token while preserving the serialized string
    /// field and JSON contract.
    pub role: &'static str,
    pub step_index: usize,
    pub token_in: u32,
    pub token_out: u32,
    pub step: Qwen38StepWall,
    pub tokenizer_decode_ns: u64,
    pub bookkeeping_ns: u64,
    pub complete_wall_ns: u64,
}

impl Qwen38CompleteToken {
    pub fn named_sum_ns(&self) -> u64 {
        self.step
            .named_sum_ns()
            .saturating_add(self.tokenizer_decode_ns)
            .saturating_add(self.bookkeeping_ns)
    }

    pub fn residual_ns(&self) -> i64 {
        self.complete_wall_ns as i64 - self.named_sum_ns() as i64
    }

    pub fn wall_minus_gpu_ns(&self) -> Option<i64> {
        Some(self.complete_wall_ns as i64 - self.step.gpu_ns? as i64)
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen38CompleteWallResult {
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub wall_ns: u64,
    pub reset_ns: u64,
    pub prefill_wall_ns: u64,
    pub decode_wall_ns: u64,
    pub fallbacks: u32,
    pub dense_w_materialized: u64,
    pub resident_weight_bytes: u64,
    pub workspace_resident_bytes: u64,
    pub steps: Vec<Qwen38CompleteToken>,
}

impl Qwen38CompleteWallResult {
    pub fn new_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len.min(self.tokens.len())..]
    }

    pub fn decode_new(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer.decode(self.new_tokens(), true)
    }

    pub fn first_step(&self) -> Option<&Qwen38CompleteToken> {
        self.steps.first()
    }

    /// Prompt walk, including the last prompt step that emits new-token[0].
    pub fn prefill_steps(&self) -> impl Iterator<Item = &Qwen38CompleteToken> {
        self.steps.iter().filter(|s| s.role != "decode")
    }

    /// New-tokens[1..]: Q80 mixed `steady_state` denominator.
    pub fn steady_decode_steps(&self) -> impl Iterator<Item = &Qwen38CompleteToken> {
        self.steps.iter().filter(|s| s.role == "decode")
    }
}

/// A resumable point in a token sequence: the DeltaNet carry plus the position
/// it was taken at. Cheap to hold (state is fixed size, independent of context)
/// and exact for any prompt that begins with the same tokens.
#[derive(Debug, Clone)]
pub struct Qwen38PrefixCheckpoint {
    pub position: usize,
    pub rec_state: Vec<f32>,
    pub conv_state: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Qwen38GenerateResult {
    /// Why generation stopped: "constraint_done", "eos" or "budget".
    ///
    /// A reply that "never closed the JSON object" has three possible causes
    /// and they need different fixes. Empty for the unconstrained path, which
    /// has no constraint to finish.
    pub stop_reason: &'static str,
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub wall_ns: u64,
    pub gpu_ns: Vec<Option<u64>>,
    pub wait_ns: Vec<u64>,
    pub encode_ns: Vec<u64>,
    pub submit_ns: Vec<u64>,
    pub dispatches: Vec<u64>,
    pub active_weight_bytes: Vec<u64>,
    pub fallbacks: u32,
    pub dense_w_materialized: u64,
    pub resident_weight_bytes: u64,
    pub workspace_resident_bytes: u64,
    pub first_step_wall_ns: u64,
    pub prefill_wall_ns: u64,
    pub decode_wall_ns: u64,
    pub decode_steps: usize,
    pub wall_ns_per_step: Vec<u64>,
}

impl Qwen38GenerateResult {
    pub fn new_tokens(&self) -> &[u32] {
        &self.tokens[self.prompt_len.min(self.tokens.len())..]
    }

    pub fn decode_new(&self, tokenizer: &Tokenizer) -> Result<String> {
        tokenizer.decode(self.new_tokens(), true)
    }

    pub fn median_gpu_ns_per_token(&self) -> Option<u64> {
        let mut values: Vec<u64> = self.gpu_ns.iter().copied().flatten().collect();
        if values.is_empty() {
            return None;
        }
        values.sort_unstable();
        Some(values[values.len() / 2])
    }

    pub fn steady_decode_wall_ns_per_token(&self) -> Option<u64> {
        if self.decode_steps == 0 {
            return None;
        }
        Some(self.decode_wall_ns / self.decode_steps as u64)
    }
}

#[cfg(target_os = "macos")]
pub use device::{
    generate_constrained, generate_greedy, generate_greedy_complete_wall, generate_greedy_parallel,
    generate_greedy_reusing, generate_greedy_reusing_snapshot, generate_greedy_unmeasured,
    measure_shared_weight_fanout, Qwen38HybridDecodeSession, Qwen38HybridWeights,
    Qwen38WeightFanout,
};

#[cfg(not(target_os = "macos"))]
pub fn generate_greedy(
    _root: impl AsRef<Path>,
    _prompt: &[u32],
    _max_new: usize,
) -> Result<Qwen38GenerateResult> {
    Err(Error::Model("qwen38 native decode is Metal-only".into()))
}

#[cfg(not(target_os = "macos"))]
pub fn generate_greedy_complete_wall(
    _root: impl AsRef<Path>,
    _tokenizer: &Tokenizer,
    _prompt: &[u32],
    _max_new: usize,
) -> Result<Qwen38CompleteWallResult> {
    Err(Error::Model("qwen38 native decode is Metal-only".into()))
}

#[cfg(test)]
mod mlp_fusion_env_tests {
    use super::*;

    /// All four assertions live in ONE test on purpose: they mutate the same
    /// process-global env var, and split across parallel `#[test]`s they would
    /// race each other.
    #[test]
    fn an_unrecognised_value_never_silently_means_off() {
        const K: &str = "HAWKING_QWEN38_FUSE_MLP";
        const FAST: &str = "HAWKING_QWEN38_FAST";
        const GQA: &str = "HAWKING_QWEN38_FUSE_GQA_QKV";
        const DN: &str = "HAWKING_QWEN38_FUSE_DN_INPROJ";
        const ADD: &str = "HAWKING_QWEN38_FUSE_ADD_RMSNORM";
        const BA: &str = "HAWKING_QWEN38_FUSE_BA_DELTA";
        const ATTENTION_GATE: &str = "HAWKING_QWEN38_FUSE_ATTENTION_GATE";
        const STATE: &str = "HAWKING_QWEN38_DN_STATE";
        const GEO: &str = "HAWKING_AFFINE2_GEO";
        const Q2F_GEO: &str = "HAWKING_Q2F_GEO";
        const Q4_GEO: &str = QWEN38_Q4_GEO_ENV;
        const SERIAL: &str = "HAWKING_QWEN38_SERIAL_TOKEN_ENCODER";
        let restore = std::env::var(K).ok();
        let restore_fast = std::env::var(FAST).ok();
        let restore_gqa = std::env::var(GQA).ok();
        let restore_dn = std::env::var(DN).ok();
        let restore_add = std::env::var(ADD).ok();
        let restore_ba = std::env::var(BA).ok();
        let restore_attention_gate = std::env::var(ATTENTION_GATE).ok();
        let restore_state = std::env::var(STATE).ok();
        let restore_geo = std::env::var(GEO).ok();
        let restore_q2f_geo = std::env::var(Q2F_GEO).ok();
        let restore_q4_geo = std::env::var(Q4_GEO).ok();
        let restore_serial = std::env::var(SERIAL).ok();
        std::env::set_var(FAST, "0");
        for key in [
            GQA,
            DN,
            ADD,
            BA,
            ATTENTION_GATE,
            STATE,
            GEO,
            Q2F_GEO,
            Q4_GEO,
            SERIAL,
        ] {
            std::env::remove_var(key);
        }

        // 1. The regression itself. `=1` is what the three sibling levers use,
        //    and it used to parse to Off -- measuring the UNFUSED graph while
        //    reporting the lever as on.
        std::env::set_var(K, "1");
        assert_eq!(
            Qwen38MlpFusion::from_env(),
            Qwen38MlpFusion::GateUpSwiglu,
            "=1 must mean the strongest fusion, as it does for every sibling lever"
        );
        assert_eq!(Qwen38MlpFusion::from_env().saved_dispatches_per_token(), 128);

        // 2. The named values still mean what the shader and the receipts say.
        std::env::set_var(K, "pair");
        assert_eq!(Qwen38MlpFusion::from_env(), Qwen38MlpFusion::GateUpPair);
        std::env::set_var(K, "swiglu");
        assert_eq!(Qwen38MlpFusion::from_env(), Qwen38MlpFusion::GateUpSwiglu);

        // 3. Off is still REACHABLE. A guard that made every value fuse would
        //    pass assertion 1 and destroy the ability to measure a baseline,
        //    which is the whole point of a default-off lever.
        std::env::set_var(K, "0");
        assert_eq!(Qwen38MlpFusion::from_env(), Qwen38MlpFusion::Off);
        std::env::remove_var(K);
        assert_eq!(Qwen38MlpFusion::from_env(), Qwen38MlpFusion::Off);

        // 4. A typo is LOUD. Without this the guard above is decoration:
        //    `=swigly` would land back in the silent-Off hole.
        std::env::set_var(K, "swigly");
        let typo = std::panic::catch_unwind(Qwen38MlpFusion::from_env);
        std::env::remove_var(K);
        assert!(typo.is_err(), "an unrecognised value must panic, not mean Off");

        // 5. The explicit fastest profile is a composition of the individually
        // measured candidates, but every named override still wins.
        std::env::set_var(FAST, "1");
        assert_eq!(Qwen38MlpFusion::from_env(), Qwen38MlpFusion::GateUpSwiglu);
        assert!(qwen38_fuse_gqa_qkv_enabled());
        assert!(qwen38_fuse_dn_inproj_enabled());
        assert_eq!(qwen38_fuse_add_rmsnorm_from_env(), (true, false));
        assert_eq!(qwen38_fuse_ba_delta_from_env(), (true, false));
        assert_eq!(Qwen38DeltaNetStateKernel::from_env(), Qwen38DeltaNetStateKernel::WidenF4);
        // G126 PROMOTED: the fast profile used to select SplitK4, which no
        // protected lease ever timed. Bitcast is the measured arm and is now the
        // default everywhere, so fast composes with it instead of overriding it.
        assert_eq!(Affine2Geo::from_env(), Affine2Geo::Bitcast);
        assert_eq!(qwen38_q2f_geo_from_env(), Affine2Geo::Bitcast);
        assert_eq!(Qwen38MatvecKernel::from_env(), Qwen38MatvecKernel::GeoTpr64Tg128);
        assert!(qwen38_serial_token_encoder_enabled());
        assert!(qwen38_fuse_attention_gate_enabled());
        std::env::set_var(GQA, "0");
        std::env::set_var(K, "0");
        std::env::set_var(STATE, "baseline");
        std::env::set_var(GEO, "tpr64");
        std::env::set_var(Q2F_GEO, "splitk4_vec");
        std::env::set_var(Q4_GEO, "vecgroup_x64");
        std::env::set_var(SERIAL, "0");
        std::env::set_var(ATTENTION_GATE, "0");
        assert!(!qwen38_fuse_gqa_qkv_enabled());
        assert_eq!(Qwen38MlpFusion::from_env(), Qwen38MlpFusion::Off);
        assert_eq!(Qwen38DeltaNetStateKernel::from_env(), Qwen38DeltaNetStateKernel::Baseline);
        assert_eq!(Affine2Geo::from_env(), Affine2Geo::Tpr64);
        assert_eq!(qwen38_q2f_geo_from_env(), Affine2Geo::SplitK4Vec);
        assert_eq!(Qwen38MatvecKernel::from_env(), Qwen38MatvecKernel::VecgroupX64);
        assert!(!qwen38_serial_token_encoder_enabled());
        assert!(!qwen38_fuse_attention_gate_enabled());

        std::env::set_var(Q4_GEO, "not-a-geometry");
        let q4_typo = std::panic::catch_unwind(Qwen38MatvecKernel::from_env);
        assert!(q4_typo.is_err(), "an unrecognised Q4 geometry must fail loudly");

        for (key, value) in [
            (K, restore),
            (FAST, restore_fast),
            (GQA, restore_gqa),
            (DN, restore_dn),
            (ADD, restore_add),
            (BA, restore_ba),
            (ATTENTION_GATE, restore_attention_gate),
            (STATE, restore_state),
            (GEO, restore_geo),
            (Q2F_GEO, restore_q2f_geo),
            (Q4_GEO, restore_q4_geo),
            (SERIAL, restore_serial),
        ] {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

#[cfg(test)]
mod dn_state_kernel_tests {
    use super::*;

    #[test]
    fn widen_f4_folds_ba_to_decay_without_the_fuse_flag() {
        // The production 628 graph is Baseline + FUSE_BA_DELTA off.
        // Selecting WidenF4 must still launch the fused-ba sibling;
        // otherwise HAWKING_QWEN38_DN_STATE=widen_f4 is a no-op and
        // encode_deltanet keeps dispatching unfused vi-SIMD.
        assert!(!Qwen38DeltaNetStateKernel::Baseline.folds_ba_to_decay());
        assert!(Qwen38DeltaNetStateKernel::WidenF4.folds_ba_to_decay());
        assert!(Qwen38DeltaNetStateKernel::CoalesceTg32.folds_ba_to_decay());
        assert!(!qwen38_dn_state_uses_fused_ba(
            false,
            Qwen38DeltaNetStateKernel::Baseline
        ));
        assert!(qwen38_dn_state_uses_fused_ba(
            false,
            Qwen38DeltaNetStateKernel::WidenF4
        ));
        assert!(qwen38_dn_state_uses_fused_ba(
            true,
            Qwen38DeltaNetStateKernel::Baseline
        ));
        assert_eq!(
            Qwen38DeltaNetStateKernel::WidenF4.fused_ba_name(false),
            QWEN38_DN_STATE_F4_KERNEL
        );
        assert_eq!(
            qwen38_fused_dispatches_per_token_full(
                Qwen38MlpFusion::GateUpSwiglu,
                true,
                true,
                true,
                qwen38_dn_state_uses_fused_ba(false, Qwen38DeltaNetStateKernel::Baseline),
            ),
            628
        );
        assert_eq!(
            qwen38_fused_dispatches_per_token_full(
                Qwen38MlpFusion::GateUpSwiglu,
                true,
                true,
                true,
                qwen38_dn_state_uses_fused_ba(false, Qwen38DeltaNetStateKernel::WidenF4),
            ),
            580
        );
    }

    #[test]
    fn dn_state_from_env_selects_widen_f4() {
        const K: &str = "HAWKING_QWEN38_DN_STATE";
        let restore = std::env::var(K).ok();
        std::env::remove_var(K);
        // G126 PROMOTED: unset is the MEASURED arm. If this ever reads Baseline
        // again the sealed graph has silently diverged from the graph the
        // protected lease timed, and every downstream absolute is stale.
        assert_eq!(
            Qwen38DeltaNetStateKernel::from_env(),
            Qwen38DeltaNetStateKernel::WidenF4
        );
        std::env::set_var(K, "widen_f4");
        assert_eq!(
            Qwen38DeltaNetStateKernel::from_env(),
            Qwen38DeltaNetStateKernel::WidenF4
        );
        std::env::set_var(K, "f4");
        assert_eq!(
            Qwen38DeltaNetStateKernel::from_env(),
            Qwen38DeltaNetStateKernel::WidenF4
        );
        std::env::set_var(K, "baseline");
        assert_eq!(
            Qwen38DeltaNetStateKernel::from_env(),
            Qwen38DeltaNetStateKernel::Baseline
        );
        if let Some(v) = restore {
            std::env::set_var(K, v);
        } else {
            std::env::remove_var(K);
        }
    }
}

#[cfg(test)]
mod mixed_catalog_contract_tests {
    use super::*;

    #[test]
    fn absolute_segment_filename_does_not_join_segments() {
        let root = Path::new("/artifact/mixed-sub15-v1");
        let abs = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-2p0-v1/segments/L00.hq38seg";
        assert_eq!(
            resolve_mixed_segment_path(root, abs),
            PathBuf::from(abs)
        );
        assert_eq!(
            resolve_mixed_segment_path(root, "L00.hq38seg"),
            root.join("segments").join("L00.hq38seg")
        );
    }

    #[test]
    fn hq38m20_magic_and_record_match_q80_layout() {
        assert_eq!(&QWEN38_MIXED_CATALOG_MAGIC, b"HQ38M20\0");
        assert_eq!(QWEN38_MIXED_RECORD_SIZE, 128);
        assert_eq!(QWEN38_MIXED_CATALOG_VERSION, 1);
        assert_eq!(
            QWEN38_MIXED_SCHEMA,
            "hawking.ascension.qwen38_mixed_representation_candidate.v1"
        );
    }

    #[test]
    fn codec_4_f32v2_is_accepted_without_mlx_delta() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u64.to_le_bytes());
        payload.extend_from_slice(&1.046875f32.to_le_bytes());
        payload.extend_from_slice(&0.5f32.to_le_bytes());
        let name = "language_model.model.layers.0.input_layernorm.weight";
        let lane = classify_qwen38_mixed_payload(4, &payload, name, &[2]).unwrap();
        assert_eq!(lane, MixedCatalogLane::F32v2);
        let values = read_qwen38_f32_payload(&payload).unwrap();
        assert_eq!(values, vec![1.046875, 0.5]);
    }

    #[test]
    fn unknown_codec_6_still_refuses() {
        let err = classify_qwen38_mixed_payload(6, b"xxxxxxxx", "tensor.x", &[1])
            .expect_err("codec 6 must refuse");
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown mixed codec 6"),
            "refuse message was {msg}"
        );
    }

    #[test]
    fn classify_codec_5_hgrafv01_is_affine() {
        let packed = super::super::qwen_complete_binary::pack_affine_factor(
            &super::super::qwen_complete_binary::deterministic_matrix(2, 32, 7),
            2,
            32,
        )
        .unwrap();
        let payload = super::super::qwen_complete_binary::wrap_affine_factor(&packed).unwrap();
        let lane = classify_qwen38_mixed_payload(5, &payload, "tensor.x", &[2, 32]).unwrap();
        assert_eq!(lane, MixedCatalogLane::Affine);
        assert_eq!(
            mixed_mlp_native_kind_from_lane(lane),
            Some(MixedMlpNativeKind::AffineScaleBias)
        );
    }

    #[test]
    fn classify_codec_5_wrong_magic_refuses() {
        let err = classify_qwen38_mixed_payload(5, b"HGRAVU01xxxx", "tensor.x", &[1])
            .expect_err("codec 5 wrong magic must refuse");
        let msg = format!("{err}");
        assert!(msg.contains("not HGRAVF01"), "refuse message was {msg}");
    }

    fn filled_mlp_kinds(kind: MixedMlpNativeKind) -> HashMap<String, MixedMlpNativeKind> {
        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(qwen38_layer_name(layer, "mlp.gate_proj.weight"), kind);
            kinds.insert(qwen38_layer_name(layer, "mlp.up_proj.weight"), kind);
            kinds.insert(qwen38_layer_name(layer, "mlp.down_proj.weight"), kind);
        }
        kinds
    }

    #[test]
    fn mixed_mlp_uniform_is_admitted_on_every_role() {
        let kinds = filled_mlp_kinds(MixedMlpNativeKind::Uniform);
        assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied()).unwrap();
    }

    #[test]
    fn mixed_mlp_affine_is_admitted_on_every_role() {
        let kinds = filled_mlp_kinds(MixedMlpNativeKind::AffineScaleBias);
        assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied()).unwrap();
    }

    #[test]
    fn mixed_mlp_all_binary_is_admitted() {
        let kinds = filled_mlp_kinds(MixedMlpNativeKind::Binary);
        assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied()).unwrap();
    }

    #[test]
    fn mixed_mlp_binary_body_affine_down_island_is_admitted() {
        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(
                qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                MixedMlpNativeKind::Binary,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.up_proj.weight"),
                MixedMlpNativeKind::Binary,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.down_proj.weight"),
                MixedMlpNativeKind::AffineScaleBias,
            );
        }
        assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied()).unwrap();
    }

    #[test]
    fn mixed_mlp_legacy_binary_residual_hgravs_still_admits() {
        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(
                qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                MixedMlpNativeKind::Binary,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.up_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.down_proj.weight"),
                MixedMlpNativeKind::Hgravs,
            );
        }
        assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied()).unwrap();
    }

    #[test]
    fn mixed_mlp_unsupported_role_still_refuses() {
        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(
                qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.up_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.down_proj.weight"),
                MixedMlpNativeKind::Hgravs,
            );
        }
        let err = assert_mixed_mlp_native_kinds(|name| kinds.get(name).copied())
            .expect_err("Residual on gate must refuse");
        let msg = format!("{err}");
        assert!(msg.contains("is not HGRAVB01"), "refuse message was {msg}");
        assert!(
            msg.contains("mlp.gate_proj.weight"),
            "refuse message was {msg}"
        );
    }

    #[test]
    fn hq30uq4_on_mlp_is_not_uniform_and_still_refuses() {
        let mut payload = b"HQ30UQ4\0".to_vec();
        payload.extend_from_slice(&[0u8; 16]);
        let name = "language_model.model.layers.0.mlp.down_proj.weight";
        let lane = classify_qwen38_mixed_payload(3, &payload, name, &[5120, 17408]).unwrap();
        assert_eq!(lane, MixedCatalogLane::Hq30Uq4);
        assert_eq!(mixed_mlp_native_kind_from_lane(lane), None);

        let mut kinds = HashMap::new();
        for layer in 0..QWEN38_LAYERS {
            kinds.insert(
                qwen38_layer_name(layer, "mlp.gate_proj.weight"),
                MixedMlpNativeKind::Binary,
            );
            kinds.insert(
                qwen38_layer_name(layer, "mlp.up_proj.weight"),
                MixedMlpNativeKind::Residual,
            );
        }
        let err = assert_mixed_mlp_native_kinds(|n| kinds.get(n).copied())
            .expect_err("HQ30UQ4 down must refuse as missing mixed native");
        let msg = format!("{err}");
        assert!(msg.contains("missing"), "refuse message was {msg}");
        assert!(
            msg.contains("mlp.down_proj.weight"),
            "refuse message was {msg}"
        );
        assert!(
            msg.contains("silent dense/Q4 fallback"),
            "refuse message was {msg}"
        );
    }

    fn campaign_qwen38(name: &str) -> PathBuf {
        PathBuf::from(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b",
        )
        .join(name)
    }

    #[test]
    fn mixed_q3mlp_and_q4down_pass_mlp_admission() {
        let q3 = campaign_qwen38("mixed-q3mlp-v1");
        let q4down = campaign_qwen38("mixed-q4down-v1");
        if !q3.join(QWEN38_MIXED_CATALOG_NAME).is_file()
            || !q4down.join(QWEN38_MIXED_CATALOG_NAME).is_file()
        {
            eprintln!("skip: mixed-q3mlp / mixed-q4down artifacts not on this host");
            return;
        }
        assert_mixed_mlp_native_catalog(&q3)
            .unwrap_or_else(|e| panic!("mixed-q3mlp-v1 must admit: {e}"));
        assert_mixed_mlp_native_catalog(&q4down)
            .unwrap_or_else(|e| panic!("mixed-q4down-v1 must admit: {e}"));
    }

    #[test]
    fn mixed_2p0_legacy_mlp_assignment_still_admits() {
        let mixed = campaign_qwen38("mixed-2p0-v1");
        if !mixed.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            eprintln!("skip: mixed-2p0-v1 artifact not on this host");
            return;
        }
        assert_mixed_mlp_native_catalog(&mixed)
            .unwrap_or_else(|e| panic!("mixed-2p0-v1 must still admit: {e}"));
    }

    #[test]
    fn k_complete_bind_retargets_wide_columns() {
        assert_eq!(
            qwen38_binary_matvec_kernel(2048),
            "q80_binary_group_matvec_tg256"
        );
        assert_eq!(
            qwen38_binary_matvec_kernel(5120),
            "q80_binary_group_matvec_simd_bytes"
        );
        assert_eq!(
            qwen38_binary_matvec_kernel(6144),
            "q80_binary_group_matvec_simd_bytes"
        );
        assert_eq!(
            qwen38_residual_matvec_kernel(5120),
            "q80_binary_group_csr_matvec_bytes"
        );
        assert_eq!(
            qwen38_residual_matvec_kernel(2048),
            "q80_binary_group_csr_matvec_tg256"
        );
        qwen38_assert_k_complete_cols(5120).unwrap();
        qwen38_assert_k_complete_cols(6144).unwrap();
        qwen38_assert_k_complete_cols(2048).unwrap();
        let err = qwen38_assert_k_complete_cols(2049).expect_err("remainder must refuse");
        assert!(format!("{err}").contains("partial-K"));
        assert!(qwen38_mixed_k_complete_bind_message().contains("K-complete"));
    }

    fn write_tiny_hq38m20(dir: &Path, name: &str, codec: u8, payload: &[u8]) {
        let seg_name = "t0.seg";
        std::fs::write(dir.join("segments").join(seg_name), payload).unwrap();
        let name_bytes = name.as_bytes();
        let mut rec = vec![0u8; QWEN38_MIXED_RECORD_SIZE];
        rec[0..4].copy_from_slice(&0u32.to_le_bytes());
        rec[4..6].copy_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        rec[6] = codec;
        rec[7] = 6;
        rec[8] = 1;
        rec[12..16].copy_from_slice(&2u32.to_le_bytes());
        rec[28..36].copy_from_slice(&2u64.to_le_bytes());
        rec[36..38].copy_from_slice(&0u16.to_le_bytes());
        rec[40..48].copy_from_slice(&0u64.to_le_bytes());
        rec[48..56].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        let mut catalog = Vec::new();
        catalog.extend_from_slice(&QWEN38_MIXED_CATALOG_MAGIC);
        catalog.extend_from_slice(&QWEN38_MIXED_CATALOG_VERSION.to_le_bytes());
        catalog.extend_from_slice(&1u32.to_le_bytes());
        catalog.extend_from_slice(&1u32.to_le_bytes());
        catalog.extend_from_slice(&0u32.to_le_bytes());
        catalog.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        catalog.extend_from_slice(&0u32.to_le_bytes());
        let digest = [0u8; 32];
        catalog.extend_from_slice(&0u16.to_le_bytes());
        catalog.extend_from_slice(&(seg_name.len() as u16).to_le_bytes());
        catalog.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        catalog.extend_from_slice(&digest);
        catalog.extend_from_slice(seg_name.as_bytes());
        catalog.extend_from_slice(&rec);
        catalog.extend_from_slice(name_bytes);
        std::fs::write(dir.join(QWEN38_MIXED_CATALOG_NAME), catalog).unwrap();
    }

    #[test]
    fn catalog_roundtrip_codec_4_census_and_codec_6_refuses() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir(root.join("segments")).unwrap();
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u64.to_le_bytes());
        payload.extend_from_slice(&0.046875f32.to_le_bytes());
        payload.extend_from_slice(&1.0f32.to_le_bytes());
        write_tiny_hq38m20(
            root,
            "language_model.model.layers.0.input_layernorm.weight",
            4,
            &payload,
        );
        let census = census_qwen38_mixed_catalog(root).unwrap();
        assert_eq!(census.tensors, 1);
        assert_eq!(census.f32, 1);
        assert_eq!(census.refused, 0);
        assert_eq!(census.expanded_to_q4, 0);
        assert_eq!(census.expanded_to_float_gemv, 0);
        assert_eq!(census.dense_w_materialized, 0);

        write_tiny_hq38m20(root, "tensor.x", 6, &payload);
        let census = census_qwen38_mixed_catalog(root).unwrap();
        assert_eq!(census.refused, 1);
        assert!(census.refusals.iter().any(|s| s.contains("unknown mixed codec 6")));
    }

    #[test]
    fn catalog_roundtrip_codec_5_affine_census() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir(root.join("segments")).unwrap();
        let packed = super::super::qwen_complete_binary::pack_affine_factor(
            &super::super::qwen_complete_binary::deterministic_matrix(2, 32, 11),
            2,
            32,
        )
        .unwrap();
        let payload = super::super::qwen_complete_binary::wrap_affine_factor(&packed).unwrap();
        write_tiny_hq38m20(
            root,
            "language_model.model.layers.0.mlp.gate_proj.weight",
            5,
            &payload,
        );
        let census = census_qwen38_mixed_catalog(root).unwrap();
        assert_eq!(census.tensors, 1);
        assert_eq!(census.affine, 1);
        assert_eq!(census.refused, 0);
        assert_eq!(census.expanded_to_q4, 0);
        assert_eq!(census.expanded_to_float_gemv, 0);
        assert_eq!(census.dense_w_materialized, 0);
    }

    #[test]
    fn sub15_source_payloads_accept_mixed_gpu_layout() {
        let camp = Path::new(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b",
        );
        let mixed = camp.join("mixed-2p0-v1");
        let sub15 = camp.join("mixed-sub15-v1");
        if !mixed.join(QWEN38_MIXED_CATALOG_NAME).is_file() || !sub15.join("packed/attn").is_dir()
        {
            eprintln!("skip: mixed-2p0 / mixed-sub15 artifacts not on this host");
            return;
        }
        let rows = parse_qwen38_mixed_catalog(&mixed).unwrap();
        for (suffix, codec) in [
            ("mlp.gate_proj.weight", 0u8),
            ("mlp.up_proj.weight", 1u8),
            ("mlp.down_proj.weight", 2u8),
        ] {
            let row = rows
                .iter()
                .find(|r| r.name.contains(".layers.0.") && r.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("missing L0 {suffix}"));
            assert_eq!(row.codec, codec);
            let payload = read_catalog_payload(row).unwrap();
            let layout = mixed_gpu_layout(codec, &payload).expect(suffix);
            assert!(layout.cols == 5120 || layout.cols == 17408, "{suffix} cols {}", layout.cols);
        }

        fn rice(name: &str) -> std::path::PathBuf {
            use sha2::{Digest, Sha256};
            let digest = Sha256::digest(name.as_bytes());
            let stem: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            PathBuf::from(format!("{stem}.rice"))
        }
        let qkv = sub15.join("packed/attn").join(rice(
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        ));
        let a = sub15.join("packed/attn").join(rice(
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
        ));
        let qkv_bytes = std::fs::read(&qkv).unwrap();
        let a_bytes = std::fs::read(&a).unwrap();
        let qkv_layout = mixed_gpu_layout(1, &qkv_bytes).expect("in_proj_qkv");
        let a_layout = mixed_gpu_layout(1, &a_bytes).expect("in_proj_a");
        assert_eq!((qkv_layout.rows, qkv_layout.cols), (10240, 5120));
        assert_eq!((a_layout.rows, a_layout.cols), (48, 5120));
    }

    #[test]
    fn sub15_native_catalog_census_if_emitted() {
        let artifact = Path::new(
            "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs/qwen38-27b/mixed-sub15-v1",
        );
        let staged = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../workspace/superwave/g1/mixed-sub15-native-catalog");
        let root = if artifact.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            artifact
        } else if staged.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            staged.as_path()
        } else {
            eprintln!("skip: mixed-sub15 catalog.hq38m20 not emitted yet");
            return;
        };
        let census = census_qwen38_mixed_catalog(root).unwrap();
        assert_eq!(census.refused, 0, "refusals: {:?}", census.refusals);
        assert_eq!(census.expanded_to_q4, 0);
        assert_eq!(census.expanded_to_float_gemv, 0);
        assert_eq!(census.dense_w_materialized, 0);
        assert_eq!(census.tensors, 851);
        assert_eq!(census.binary, 64);
        assert_eq!(census.residual, 368);
        assert_eq!(census.hgravs, 64);
        assert_eq!(census.uniform, 0);
        assert_eq!(census.q4, 2);
        assert_eq!(census.f32, 353);
    }

    #[test]
    fn hgravu01_geo_tpr64_bind_is_bits_3_and_4_only() {
        let q3 = qwen38_hgravu01_geo_tpr64_launch(3, 64, 17408, 5120).expect("q3");
        assert_eq!(q3.0, QWEN38_HGRAVU01_Q3_GEO_TPR64);
        assert_eq!(q3.1, (17408u32.div_ceil(2) * 128, 1, 1));
        assert_eq!(q3.2, (128, 1, 1));
        let q4 = qwen38_hgravu01_geo_tpr64_launch(4, 64, 48, 5120).expect("q4");
        assert_eq!(q4.0, QWEN38_HGRAVU01_Q4_GEO_TPR64);
        assert_eq!(q4.1, (48u32.div_ceil(2) * 128, 1, 1));
        let q3_g128 = qwen38_hgravu01_geo_tpr64_launch(3, 128, 12288, 5120).expect("q3g128");
        assert_eq!(q3_g128.0, QWEN38_HGRAVU01_Q3_G128_GEO_TPR64);
        assert_eq!(q3_g128.1, (12288u32.div_ceil(2) * 128, 1, 1));
        assert_eq!(q3_g128.2, (128, 1, 1));
        let lm_g128 = qwen38_hgravu01_geo_tpr64_launch(3, 128, 248320, 5120).expect("lm_g128");
        assert_eq!(lm_g128.0, QWEN38_HGRAVU01_Q3_G128_GEO_TPR64);
        assert!(qwen38_hgravu01_geo_tpr64_launch(8, 64, 5120, 5120).is_none());
        assert!(qwen38_hgravu01_geo_tpr64_launch(3, 64, 2048, 160).is_none());
        assert!(qwen38_hgravu01_geo_tpr64_launch(3, 128, 100, 192).is_none());
        assert!(qwen38_hgravu01_geo_tpr64_launch(4, 128, 248320, 5120).is_none());
        assert!(qwen38_hgravu01_geo_tpr64_launch(4, 32, 5120, 5120).is_none());
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_uniform_q3_group64_matvec_geo_tpr64_tg128"));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_uniform_q3_group128_matvec_geo_tpr64_tg128"));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_uniform_hgravu_q4_group64_matvec_geo_tpr64_tg128"));

        // HQ30UQ4 supported set is exactly {64, 128}. An unsupported group
        // size still refuses — a gate that stops refusing is not a fixed gate.
        let hq64 = qwen38_uniform_q4_geo_tpr64_launch(64, 248320, 5120).expect("hq64");
        // The bind under test is the FAMILY, not the unpack arm: G126 flipped
        // the q4 default to bitcast, and a bare const here would be asserting
        // the ambient default rather than the catalog contract.
        assert_eq!(hq64.0, qwen38_q4_kernel(QWEN38_Q4_MATVEC_KERNEL));
        assert_eq!(hq64.1, (248320u32.div_ceil(2) * 128, 1, 1));
        assert_eq!(hq64.2, (128, 1, 1));
        let hq128 = qwen38_uniform_q4_geo_tpr64_launch(128, 248320, 5120).expect("hq128");
        assert_eq!(hq128.0, QWEN38_Q4_GROUP128_MATVEC_KERNEL);
        assert_eq!(hq128.1, (248320u32.div_ceil(2) * 128, 1, 1));
        assert_eq!(hq128.2, (128, 1, 1));
        for group in [0u32, 32, 96, 256, 512] {
            assert!(
                qwen38_uniform_q4_geo_tpr64_launch(group, 5120, 5120).is_none(),
                "HQ30UQ4 group {group} must refuse"
            );
        }
        assert!(qwen38_uniform_q4_geo_tpr64_launch(128, 5120, 160).is_none());
        assert!(qwen38_uniform_q4_geo_tpr64_launch(64, 5120, 160).is_none());
        assert!(crate::metal::SHADER_QWEN_UNIFORM_Q4
            .contains("kernel void qwen_uniform_q4_group128_matvec_geo_tpr64_tg128("));
    }

    #[test]
    fn affine_q2_geo_tpr64_bind_is_group32_or_64() {
        let on32 = qwen38_affine_q2_geo_tpr64_launch(32, 17408, 5120);
        let on64 = qwen38_affine_q2_geo_tpr64_launch(64, 17408, 5120);
        if qwen38_recon_fuse_enabled() {
            let launch = on32.expect("affine geo g32");
            assert_eq!(launch.0, QWEN38_AFFINE_Q2_GEO_TPR64);
            assert_eq!(launch.1, (17408u32.div_ceil(2) * 128, 1, 1));
            assert_eq!(launch.2, (128, 1, 1));
            let launch64 = on64.expect("affine geo g64");
            assert_eq!(launch64.0, QWEN38_AFFINE_Q2_GEO_TPR64);
        } else {
            assert!(on32.is_none());
            assert!(on64.is_none());
        }
        assert!(qwen38_affine_q2_geo_tpr64_launch(16, 17408, 5120).is_none());
        assert!(qwen38_affine_q2_geo_tpr64_launch(32, 17408, 161).is_none());
        assert!(qwen38_affine_q2_geo_tpr64_launch(64, 17408, 161).is_none());
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128"));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group32_matvec("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128_runtime_div("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains("const uint group = col >> 6u;"));
        assert_eq!(
            QWEN38_AFFINE_GATE_UP_KERNEL,
            "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128"
        );
        assert_eq!(
            QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL,
            "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128"
        );
        assert_eq!(
            QWEN38_AFFINE_Q2_GEO_TPR64_RUNTIME_DIV,
            "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_runtime_div"
        );
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_qmvfast_r8tg64("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_wide64_r4tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_tgx_r8tg256("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_qmvfast_r8tg64("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_tgsb_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_pipe_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_splitk4_tg256("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_affine_q2_group64_matvec_accfuse_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_tgsb_tpr64_tg128("
        ));
        assert!(crate::metal::SHADER_QWEN38_DEVICE_ACTIVATIONS
            .contains("kernel void qwen38_gated_delta_decode_vi_simd_ba("));
        assert!(crate::metal::SHADER_QWEN38_DEVICE_ACTIVATIONS
            .contains("kernel void qwen38_gated_delta_decode_vi_simd_ba_plain("));
        assert!(crate::metal::SHADER_QWEN38_DEVICE_ACTIVATIONS
            .contains("kernel void qwen38_gated_delta_decode_vi_simd_ba_f4("));
        assert!(crate::metal::SHADER_QWEN38_DEVICE_ACTIVATIONS
            .contains("kernel void qwen38_gated_delta_decode_vi_simd_ba_tg32("));
        assert_eq!(
            qwen38_fused_dispatches_per_token_full(
                Qwen38MlpFusion::GateUpSwiglu,
                true,
                true,
                true,
                true
            ),
            580
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::QmvFast, 64, 17408, 5120)
                .map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_QMVFAST)
        );
        assert!(qwen38_affine_q2_launch(Affine2Geo::QmvFast, 32, 17408, 5120).is_none());
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::Tgsb, 64, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_TGSB)
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::Pipe, 64, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_PIPE)
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::SplitK4, 64, 17408, 5120).map(|l| (l.0, l.2)),
            Some((QWEN38_AFFINE_Q2_SPLITK4, (256, 1, 1)))
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::SplitK4Vec, 64, 17408, 5120)
                .map(|l| (l.0, l.2)),
            Some((QWEN38_AFFINE_Q2_SPLITK4_VEC, (256, 1, 1)))
        );
        assert_eq!(
            qwen38_affine_gate_up_launch(Affine2Geo::SplitK4Vec, false, 17408).0,
            QWEN38_AFFINE_GATE_UP_SPLITK4_VEC
        );
        assert_eq!(
            qwen38_affine_gate_up_launch(Affine2Geo::SplitK4Vec, true, 17408).0,
            QWEN38_AFFINE_GATE_UP_SWIGLU_SPLITK4_VEC
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::AccFuse, 64, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_ACCFUSE)
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::FoldAddqx, 64, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_FOLD_ADDQX)
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::FoldAddqx, 32, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_FOLD_ADDQX)
        );
        assert_eq!(
            qwen38_affine_gate_up_launch(Affine2Geo::FoldAddqx, true, 17408).0,
            QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX
        );
        assert_eq!(
            qwen38_affine_gate_up_launch(Affine2Geo::FoldAddqx, false, 17408).0,
            QWEN38_AFFINE_GATE_UP_FOLD_ADDQX
        );
        // bitcast: the convert-free unpack. Selected only when asked for, and
        // the unfused gate_up form falls back to production because that kernel
        // was deliberately not written - the resident never dispatches it.
        assert_eq!(
            Affine2Geo::from_value("bitcast"),
            Affine2Geo::Bitcast
        );
        assert_eq!(Affine2Geo::Bitcast.as_str(), "bitcast");
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::Bitcast, 64, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_BITCAST)
        );
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::Bitcast, 32, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_BITCAST)
        );
        assert_eq!(
            qwen38_affine_gate_up_launch(Affine2Geo::Bitcast, true, 17408).0,
            QWEN38_AFFINE_GATE_UP_SWIGLU_BITCAST
        );
        assert_eq!(
            qwen38_affine_gate_up_launch(Affine2Geo::Bitcast, false, 17408).0,
            QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL,
            "the unfused bitcast kernel does not exist; naming it would bind a \
             pipeline that fails to compile at launch"
        );
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128_bitcast("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_bitcast("
        ));
        // The refold is only correct if the mantissa step is 0.5 per code.
        // A 0.25 step compiles, runs 1.2x, and returns garbage.
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("w = (2*scale)*f + (bias - 4*scale)"));

        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group32_matvec_geo_tpr64_tg128_fold_addqx("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_fold_addqx("
        ));
        assert_eq!(Affine2Geo::from_value("fold_addqx").as_str(), "fold_addqx");
        assert_eq!(
            qwen38_affine_q2_launch(Affine2Geo::BiasPrep, 64, 17408, 5120).map(|l| l.0),
            Some(QWEN38_AFFINE_Q2_GEO_TPR64)
        );
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_splitk4_vec_tg256("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_splitk4_vec_tg256("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_affine_q2_group64_matvec_gate_up_swiglu_splitk4_vec_tg256("
        ));
        assert!(crate::metal::SHADER_QWEN80_DEVICE_ACTIVATIONS
            .contains("kernel void qwen80_residual_rmsnorm_tg_xsum64("));
        assert!(crate::metal::SHADER_QWEN80_DEVICE_ACTIVATIONS
            .contains("kernel void qwen80_add_residual_rmsnorm_tg_xsum64("));
        assert_eq!(
            qwen38_affine_gate_up_launch(Affine2Geo::BiasPrep, true, 17408).0,
            QWEN38_AFFINE_GATE_UP_SWIGLU_BIASPREP
        );
        assert!(qwen38_affine_q2_launch(Affine2Geo::Tgsb, 32, 17408, 5120).is_none());
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_q2f_group64_matvec_geo_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_q2f_group64_matvec_qkv_geo_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_q2f_group64_matvec_pair_geo_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_q2f_group64_matvec("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("(float(q) - 1.5f) * delta"));
        assert_eq!(QWEN38_Q2F_GEO_TPR64, "qwen_q2f_group64_matvec_geo_tpr64_tg128");
        assert_eq!(
            QWEN38_Q2F_QKV_GEO_KERNEL,
            "qwen_q2f_group64_matvec_qkv_geo_tpr64_tg128"
        );
        assert_eq!(
            QWEN38_Q2F_PAIR_GEO_KERNEL,
            "qwen_q2f_group64_matvec_pair_geo_tpr64_tg128"
        );
        assert_eq!(
            QWEN38_Q2F_GATE_UP_KERNEL,
            "qwen_q2f_group64_matvec_gate_up_geo_tpr64_tg128"
        );
        assert_eq!(
            qwen38_q2f_matvec_launch(Affine2Geo::Pipe, 17408).0,
            QWEN38_Q2F_PIPE
        );
        assert_eq!(
            qwen38_q2f_matvec_launch(Affine2Geo::SplitK4, 17408).0,
            QWEN38_Q2F_SPLITK4
        );
        assert_eq!(
            qwen38_q2f_gate_up_launch(Affine2Geo::Pipe, true, 17408).0,
            QWEN38_Q2F_GATE_UP_SWIGLU_PIPE
        );
        assert_eq!(
            qwen38_q2f_gate_up_launch(Affine2Geo::SplitK4, false, 17408).0,
            QWEN38_Q2F_GATE_UP_SPLITK4
        );
        assert_eq!(
            qwen38_q2f_gate_up_launch(Affine2Geo::SplitK4, true, 17408).0,
            QWEN38_Q2F_GATE_UP_SWIGLU_SPLITK4
        );
        assert_eq!(
            qwen38_q2f_matvec_launch(Affine2Geo::SplitK4Vec, 17408).0,
            QWEN38_Q2F_SPLITK4_VEC
        );
        assert_eq!(
            qwen38_q2f_gate_up_launch(Affine2Geo::SplitK4Vec, false, 17408).0,
            QWEN38_Q2F_GATE_UP_SPLITK4_VEC
        );
        assert_eq!(
            qwen38_q2f_gate_up_launch(Affine2Geo::SplitK4Vec, true, 17408).0,
            QWEN38_Q2F_GATE_UP_SWIGLU_SPLITK4_VEC
        );
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_q2f_group64_matvec_pipe_tpr64_tg128("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_q2f_group64_matvec_splitk4_tg256("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE
            .contains("kernel void qwen_q2f_group64_matvec_splitk4_vec_tg256("));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_q2f_group64_matvec_gate_up_swiglu_pipe_tpr64_tg128("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_q2f_group64_matvec_gate_up_swiglu_splitk4_tg256("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_q2f_group64_matvec_gate_up_splitk4_vec_tg256("
        ));
        assert!(crate::metal::SHADER_Q80_MIXED_DECODE.contains(
            "kernel void qwen_q2f_group64_matvec_gate_up_swiglu_splitk4_vec_tg256("
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hgravu01_geo_tpr64_matches_incumbent_on_real_tensors() {
        let root = campaign_qwen38("mixed-q3mlp-v1");
        if !root.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
            eprintln!("skip: mixed-q3mlp-v1 not on this host");
            return;
        }
        let context = match crate::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skip: MetalContext::new failed: {err}");
                return;
            }
        };
        let rows = parse_qwen38_mixed_catalog(&root).unwrap();
        let wanted = [
            "language_model.model.layers.0.mlp.gate_proj.weight",
            "language_model.model.layers.0.mlp.up_proj.weight",
            "language_model.model.layers.0.mlp.down_proj.weight",
            "language_model.model.layers.0.linear_attn.in_proj_a.weight",
            "language_model.model.layers.0.linear_attn.in_proj_b.weight",
            "language_model.model.layers.0.linear_attn.in_proj_z.weight",
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            "language_model.model.layers.0.linear_attn.out_proj.weight",
            "language_model.model.layers.3.self_attn.q_proj.weight",
            "language_model.model.layers.3.self_attn.k_proj.weight",
            "language_model.model.layers.3.self_attn.v_proj.weight",
            "language_model.model.layers.3.self_attn.o_proj.weight",
            "language_model.lm_head.weight",
        ];
        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        let mut worst_name = "";
        eprintln!(
            "PARITY_TABLE header: name bits rows cols incumbent max_abs max_rel rms n_abs_gt_1e4 n_abs_gt_1e2"
        );
        for name in wanted {
            let row = rows
                .iter()
                .find(|r| r.name == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let payload = read_catalog_payload(row).unwrap();
            let layout = mixed_gpu_layout(3, &payload).unwrap_or_else(|e| panic!("{name}: {e}"));
            let MixedGpuKind::Uniform(factor) = layout.kind else {
                panic!("{name} is not HGRAVU01 Uniform");
            };
            let incumbent = if factor.bits == 3 {
                "q80_hgravs01_factor_matvec_simd3"
            } else if factor.bits == 4 {
                "q80_hgravs01_factor_matvec_simd"
            } else {
                panic!("{name} bits={} not 3 or 4", factor.bits);
            };
            let (geo_name, geo_grid, geo_tg) = qwen38_hgravu01_geo_tpr64_launch(
                factor.bits,
                factor.group_size,
                factor.rows,
                factor.cols,
            )
            .unwrap_or_else(|| panic!("{name} refused geo bind"));
            let codes = context
                .new_buffer_with_bytes_checked(
                    &payload[factor.code_off..factor.code_off + factor.code_bytes],
                )
                .unwrap();
            let scales = context
                .new_buffer_with_bytes_checked(
                    &payload[factor.scale_off..factor.scale_off + factor.scale_bytes],
                )
                .unwrap();
            let mut x = vec![0.0f32; factor.cols as usize];
            for (i, slot) in x.iter_mut().enumerate() {
                *slot = (i % 17) as f32 * 0.125 - 1.0;
            }
            let input = context
                .new_buffer_with_bytes_checked(bytemuck::cast_slice(&x))
                .unwrap();
            let out_inc = context
                .new_buffer_checked(factor.rows as usize * 4)
                .unwrap();
            let out_geo = context
                .new_buffer_checked(factor.rows as usize * 4)
                .unwrap();
            let bind = |enc: &metal::ComputeCommandEncoderRef, out: &metal::Buffer| {
                enc.set_buffer(0, Some(&codes), 0);
                enc.set_buffer(1, Some(&scales), 0);
                enc.set_buffer(2, Some(&input), 0);
                enc.set_buffer(3, Some(out), 0);
                for (index, value) in [
                    (4u64, factor.rows),
                    (5, factor.cols),
                    (6, factor.group_size),
                    (7, factor.bits),
                    (8, factor.bound),
                ] {
                    enc.set_bytes(index, 4, &value as *const u32 as *const _);
                }
            };
            let inc_grid = (
                factor.rows.div_ceil(8).saturating_mul(256).max(256),
                1,
                1,
            );
            let mut tcb = crate::metal::TokenCommandBuffer::new(&context);
            tcb.dispatch_threads(incumbent, inc_grid, (256, 1, 1), |enc| {
                bind(enc, &out_inc)
            })
            .unwrap();
            tcb.dispatch_threads(geo_name, geo_grid, geo_tg, |enc| {
                bind(enc, &out_geo)
            })
            .unwrap();
            tcb.commit_and_wait().unwrap();
            let n = factor.rows as usize;
            let inc = unsafe {
                std::slice::from_raw_parts(out_inc.contents() as *const f32, n)
            };
            let geo = unsafe {
                std::slice::from_raw_parts(out_geo.contents() as *const f32, n)
            };
            let inc_nonzero = inc.iter().filter(|v| v.abs() > 0.0).count();
            let geo_nonzero = geo.iter().filter(|v| v.abs() > 0.0).count();
            let inc_max = inc.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let geo_max = geo.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            eprintln!(
                "PARITY_SAMPLE {name} codes={} scales={} inc_nz={inc_nonzero}/{n} geo_nz={geo_nonzero}/{n} inc_max={inc_max:.6e} geo_max={geo_max:.6e} inc[0..4]={:?} geo[0..4]={:?}",
                factor.code_bytes,
                factor.scale_bytes,
                &inc[..4.min(n)],
                &geo[..4.min(n)]
            );
            assert!(
                inc_nonzero > n / 2 && geo_nonzero > n / 2,
                "{name} output looks dead (inc_nz={inc_nonzero} geo_nz={geo_nonzero} n={n})"
            );
            if n <= 48 {
                let scales = &payload[factor.scale_off..factor.scale_off + factor.scale_bytes];
                let mut scales_f16 = Vec::with_capacity(factor.scale_bytes / 2);
                for chunk in scales.chunks_exact(2) {
                    scales_f16.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                let packed = UniformFactorPacked {
                    rows: factor.rows as usize,
                    cols: factor.cols as usize,
                    bits: u8::try_from(factor.bits).unwrap(),
                    group_size: factor.group_size as usize,
                    groups: (factor.rows as usize * factor.cols as usize)
                        .div_ceil(factor.group_size as usize),
                    bound: u16::try_from(factor.bound).unwrap(),
                    scales_f16,
                    codes: payload[factor.code_off..factor.code_off + factor.code_bytes].to_vec(),
                };
                let mut cpu_max = 0.0f32;
                for r in 0..packed.rows {
                    let mut sum = 0.0f32;
                    for c in 0..packed.cols {
                        sum += uniform_factor_value(&packed, r, c) * x[c];
                    }
                    let d = (geo[r] - sum).abs();
                    if d > cpu_max {
                        cpu_max = d;
                    }
                }
                eprintln!("PARITY_CPU {name} max_abs_vs_serial={cpu_max:.8e}");
                assert!(cpu_max <= 1.0e-2, "{name} vs CPU serial max_abs={cpu_max}");
            }
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            let mut sumsq = 0.0f64;
            let mut n_1e4 = 0usize;
            let mut n_1e2 = 0usize;
            for i in 0..n {
                let d = (geo[i] - inc[i]).abs();
                let denom = inc[i].abs().max(1.0);
                let rel = d / denom;
                if d > max_abs {
                    max_abs = d;
                }
                if rel > max_rel {
                    max_rel = rel;
                }
                sumsq += f64::from(d) * f64::from(d);
                if d > 1.0e-4 {
                    n_1e4 += 1;
                }
                if d > 1.0e-2 {
                    n_1e2 += 1;
                }
            }
            let rms = (sumsq / n as f64).sqrt();
            eprintln!(
                "PARITY {name} bits={} {}x{} {incumbent} max_abs={max_abs:.8e} max_rel={max_rel:.8e} rms={rms:.8e} n>1e-4={n_1e4} n>1e-2={n_1e2}",
                factor.bits, factor.rows, factor.cols
            );
            if max_abs > worst_abs {
                worst_abs = max_abs;
                worst_name = name;
            }
            if max_rel > worst_rel {
                worst_rel = max_rel;
            }
            let mut first_bad = None;
            let mut last_bad = 0usize;
            if n_1e2 > 0 {
                let mut even_bad = 0usize;
                let mut odd_bad = 0usize;
                let mut samples = Vec::new();
                for i in 0..n {
                    if (geo[i] - inc[i]).abs() > 1.0e-2 {
                        if first_bad.is_none() {
                            first_bad = Some(i);
                        }
                        last_bad = i;
                        if i % 2 == 0 {
                            even_bad += 1;
                        } else {
                            odd_bad += 1;
                        }
                        if samples.len() < 8 {
                            samples.push((i, inc[i], geo[i]));
                        }
                    }
                }
                eprintln!(
                    "PARITY_BAD {name} first={first_bad:?} last={last_bad} even={even_bad} odd={odd_bad} samples={samples:?} n={n}"
                );
                if n > 48 {
                    let scales = &payload[factor.scale_off..factor.scale_off + factor.scale_bytes];
                    let mut scales_f16 = Vec::with_capacity(factor.scale_bytes / 2);
                    for chunk in scales.chunks_exact(2) {
                        scales_f16.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                    }
                    let packed = UniformFactorPacked {
                        rows: factor.rows as usize,
                        cols: factor.cols as usize,
                        bits: u8::try_from(factor.bits).unwrap(),
                        group_size: factor.group_size as usize,
                        groups: (factor.rows as usize * factor.cols as usize)
                            .div_ceil(factor.group_size as usize),
                        bound: u16::try_from(factor.bound).unwrap(),
                        scales_f16,
                        codes: payload[factor.code_off..factor.code_off + factor.code_bytes]
                            .to_vec(),
                    };
                    for &(r, inc_v, geo_v) in samples.iter().take(3) {
                        let mut sum = 0.0f32;
                        for c in 0..packed.cols {
                            sum += uniform_factor_value(&packed, r, c) * x[c];
                        }
                        let d_inc = (inc_v - sum).abs();
                        let d_geo = (geo_v - sum).abs();
                        eprintln!(
                            "PARITY_CPU_ROW {name} row={r} cpu={sum:.8e} inc={inc_v:.8e} geo={geo_v:.8e} d_inc={d_inc:.3e} d_geo={d_geo:.3e}"
                        );
                        assert_eq!(d_geo, 0.0, "{name} row {r} geo must match CPU serial");
                        assert_eq!(
                            d_inc, 0.0,
                            "{name} row {r} incumbent must match CPU serial after source extract fix"
                        );
                    }
                    if let Some(r0) = first_bad {
                        if r0 > 0 {
                            let mut sum = 0.0f32;
                            for c in 0..packed.cols {
                                sum += uniform_factor_value(&packed, r0 - 1, c) * x[c];
                            }
                            eprintln!(
                                "PARITY_CPU_ROW {name} row={} cpu={sum:.8e} inc={:.8e} geo={:.8e}",
                                r0 - 1,
                                inc[r0 - 1],
                                geo[r0 - 1]
                            );
                        }
                    }
                }
            }
            if name.ends_with("lm_head.weight") {
                // Pre-fix: incumbent `bit0 = element * bits` wrapped at
                // element >= 2^32/bits (row 209715 at bits=4, K=5120) and
                // this block required first_bad == that row, last_bad == n-1,
                // n_1e2 > 30_000. Source extract is now overflow-safe, so
                // incumbent, geo, and the CPU serial oracle agree on the tail.
                let overflow_el = (1u64 << 32) / u64::from(factor.bits);
                let first_overflow_row = (overflow_el / u64::from(factor.cols)) as usize;
                assert_eq!(first_overflow_row, 209715, "lm_head wrap row (bits=4)");
                assert_eq!(
                    first_bad, None,
                    "lm_head incumbent must match geo after source extract fix"
                );
                assert_eq!(n_1e2, 0, "lm_head must have no |d|>1e-2 rows");
                assert_eq!(max_abs, 0.0, "lm_head must be bit-identical to geo");
                let _ = last_bad;
            } else {
                assert_eq!(
                    max_abs, 0.0,
                    "{name} max_abs={max_abs} is not bit-identical to incumbent"
                );
                assert_eq!(n_1e2, 0, "{name} has {n_1e2} rows with |d|>1e-2");
            }
        }
        eprintln!(
            "PARITY_WORST tensor={worst_name} max_abs={worst_abs:.8e} max_rel={worst_rel:.8e}"
        );
    }

    fn overflowing_incumbent_kernel(bits: u32) -> &'static str {
        // Always the extract_wide / extract simd family. Never geo_tpr64
        // (bits 3/4) and never q80_uniform8 (bits 8) — those paths hide the
        // wrap that the source extract still performs.
        if bits == 3 {
            "q80_hgravs01_factor_matvec_simd3"
        } else {
            "q80_hgravs01_factor_matvec_simd"
        }
    }

    fn uint32_first_overflow_row(bits: u32, cols: u32) -> usize {
        let wrap_el = (1u64 << 32) / u64::from(bits);
        (wrap_el / u64::from(cols)) as usize
    }

    fn packed_from_uniform_payload(
        factor: &crate::model::qwen_complete_binary::MixedFactorLayout,
        payload: &[u8],
    ) -> UniformFactorPacked {
        let scales = &payload[factor.scale_off..factor.scale_off + factor.scale_bytes];
        let mut scales_f16 = Vec::with_capacity(factor.scale_bytes / 2);
        for chunk in scales.chunks_exact(2) {
            scales_f16.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        UniformFactorPacked {
            rows: factor.rows as usize,
            cols: factor.cols as usize,
            bits: u8::try_from(factor.bits).unwrap(),
            group_size: factor.group_size as usize,
            groups: (factor.rows as usize * factor.cols as usize)
                .div_ceil(factor.group_size as usize),
            bound: u16::try_from(factor.bound).unwrap(),
            scales_f16,
            codes: payload[factor.code_off..factor.code_off + factor.code_bytes].to_vec(),
        }
    }

    #[test]
    fn g0_uniform_q4_geo_tpr64_source_is_unchanged() {
        let src = crate::metal::SHADER_QWEN_UNIFORM_Q4;
        assert!(src.contains("kernel void qwen_uniform_q4_group64_matvec_geo_tpr64_tg128("));
        assert!(src.contains("kernel void qwen_uniform_q4_group128_matvec_geo_tpr64_tg128("));
        assert!(src.contains("kernel void qwen_uniform_q4_group64_matvec_gate_up_geo_tpr64_tg128("));
        assert!(src.contains(
            "kernel void qwen_uniform_q4_group64_matvec_gate_up_swiglu_geo_tpr64_tg128("
        ));
        assert!(src.contains(
            "kernel void qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128("
        ));
        assert!(src.contains("kernel void qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128("));
        assert!(
            !src.contains("element * bits"),
            "G0 Q4 kernel must not use the overflowing element*bits extract"
        );
        assert!(src.contains("int(nibble) - 8"));
        assert!(src.contains("const uint group = col / QWEN_UNIFORM_Q4_GROUP_SIZE;"));
        assert!(src.contains(
            "const uint packed = *((device const uint*)(codes + rgb * QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP + (local >> 1u)));"
        ));
        assert_eq!(
            QWEN38_Q4_MATVEC_KERNEL,
            "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"
        );
        assert_eq!(
            QWEN38_Q4_GROUP128_MATVEC_KERNEL,
            "qwen_uniform_q4_group128_matvec_geo_tpr64_tg128"
        );
        assert!(
            !crate::metal::SHADER_GK_FAMILY.contains("const uint bit0 = element * bits"),
            "source extract must not form element*bits in uint32"
        );
        assert!(crate::metal::SHADER_GK_FAMILY.contains("gk_packed_lsb_byte"));
        assert_eq!(
            qwen38_hgravu01_geo_tpr64_launch(4, 64, 248320, 5120)
                .map(|(name, _, _)| name),
            Some(QWEN38_HGRAVU01_Q4_GEO_TPR64),
            "HGRAVU bits=4 still binds geo; G0 HQ30UQ4 bind is a different kernel"
        );
        assert_eq!(
            qwen38_uniform_q4_geo_tpr64_launch(64, 248320, 5120)
                .map(|(name, _, _)| name),
            Some(qwen38_q4_kernel(QWEN38_Q4_MATVEC_KERNEL))
        );
    }

    fn hq30uq4_cpu_row(
        codes: &[u8],
        scales: &[u16],
        group_size: usize,
        row: usize,
        cols: usize,
        x: &[f32],
    ) -> f32 {
        let groups_per_row = cols / group_size;
        let code_bytes = group_size / 2;
        let mut sum = 0.0f32;
        for col in 0..cols {
            let group = col / group_size;
            let local = col % group_size;
            let rgb = row * groups_per_row + group;
            let packed = codes[rgb * code_bytes + local / 2];
            let nibble = if local & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale = half::f16::from_bits(scales[rgb]).to_f32();
            sum += (i32::from(nibble) - 8) as f32 * scale * x[col];
        }
        sum
    }

    fn patterned_hq30uq4_planes(
        rows: usize,
        cols: usize,
        group_size: usize,
        fill_rows: &[usize],
    ) -> (Vec<u16>, Vec<u8>) {
        let groups_per_row = cols / group_size;
        let groups = rows * groups_per_row;
        let code_bytes = group_size / 2;
        let mut scales = vec![0u16; groups];
        let mut codes = vec![0u8; groups * code_bytes];
        let one = half::f16::from_f32(1.0).to_bits();
        for &row in fill_rows {
            for group in 0..groups_per_row {
                let rgb = row * groups_per_row + group;
                scales[rgb] = one;
                for local in 0..group_size {
                    let nibble = ((row + group + local) & 0x0f) as u8;
                    let byte = rgb * code_bytes + local / 2;
                    if local & 1 == 0 {
                        codes[byte] |= nibble;
                    } else {
                        codes[byte] |= nibble << 4;
                    }
                }
            }
        }
        (scales, codes)
    }

    fn ramp_x(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|i| (i % 17) as f32 * 0.125 - 1.0)
            .collect()
    }

    #[cfg(target_os = "macos")]
    fn dispatch_hq30uq4_geo(
        context: &crate::metal::MetalContext,
        kernel: &str,
        grid: (u32, u32, u32),
        tg: (u32, u32, u32),
        codes: &[u8],
        scales: &[u16],
        x: &[f32],
        rows: u32,
        cols: u32,
        groups_per_row: u32,
    ) -> Vec<f32> {
        let codes_b = context.new_buffer_with_bytes_checked(codes).unwrap();
        let scales_b = context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(scales))
            .unwrap();
        let input = context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(x))
            .unwrap();
        let output = context.new_buffer_checked(rows as usize * 4).unwrap();
        let mut tcb = crate::metal::TokenCommandBuffer::new(context);
        tcb.dispatch_threads(kernel, grid, tg, |enc| {
            enc.set_buffer(0, Some(&codes_b), 0);
            enc.set_buffer(1, Some(&scales_b), 0);
            enc.set_buffer(2, Some(&input), 0);
            enc.set_buffer(3, Some(&output), 0);
            for (index, value) in [(4u64, rows), (5, cols), (6, groups_per_row)] {
                enc.set_bytes(index, 4, &value as *const u32 as *const _);
            }
        })
        .unwrap();
        tcb.commit_and_wait().unwrap();
        let n = rows as usize;
        unsafe { std::slice::from_raw_parts(output.contents() as *const f32, n) }.to_vec()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hq30uq4_group64_geo_matches_serial_and_cpu() {
        let context = match crate::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skip: MetalContext::new failed: {err}");
                return;
            }
        };
        const ROWS: usize = 32;
        const COLS: usize = 256;
        let values: Vec<f32> = (0..ROWS * COLS)
            .map(|i| (i % 19) as f32 * 0.0625 - 0.5)
            .collect();
        let (payload, _) =
            crate::model::qwen_complete_binary::pack_uniform_q4_group64(&values, &[ROWS, COLS])
                .unwrap();
        let header = parse_uniform_q4_header(&payload).unwrap();
        assert_eq!(header.group_size, 64);
        let scales: Vec<u16> = payload[header.scale_offset..header.sign_offset]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let codes = payload[header.sign_offset..header.payload_bytes].to_vec();
        let x = ramp_x(COLS);
        let (name, grid, tg) =
            qwen38_uniform_q4_geo_tpr64_launch(64, ROWS as u32, COLS as u32).expect("g64 bind");
        assert_eq!(name, qwen38_q4_kernel(QWEN38_Q4_MATVEC_KERNEL));
        let geo = dispatch_hq30uq4_geo(
            &context,
            name,
            grid,
            tg,
            &codes,
            &scales,
            &x,
            ROWS as u32,
            COLS as u32,
            (COLS / 64) as u32,
        );
        let serial = dispatch_hq30uq4_geo(
            &context,
            "qwen_uniform_q4_group64_matvec",
            (ROWS as u32, 1, 1),
            (256, 1, 1),
            &codes,
            &scales,
            &x,
            ROWS as u32,
            COLS as u32,
            (COLS / 64) as u32,
        );
        let mut max_geo_serial = 0.0f32;
        let mut max_geo_cpu = 0.0f32;
        for row in 0..ROWS {
            let cpu = hq30uq4_cpu_row(&codes, &scales, 64, row, COLS, &x);
            let d_serial = (geo[row] - serial[row]).abs();
            let d_cpu = (geo[row] - cpu).abs();
            max_geo_serial = max_geo_serial.max(d_serial);
            max_geo_cpu = max_geo_cpu.max(d_cpu);
            eprintln!(
                "G64_BITIDENT row={row} cpu={cpu:.8e} serial={:.8e} geo={:.8e} d_serial={d_serial:.3e} d_cpu={d_cpu:.3e}",
                serial[row], geo[row]
            );
            assert_eq!(d_serial, 0.0, "g64 geo vs serial row {row}");
            assert_eq!(d_cpu, 0.0, "g64 geo vs CPU row {row}");
        }
        eprintln!(
            "G64_BITIDENT_SUMMARY rows={ROWS} cols={COLS} max_geo_serial={max_geo_serial:.8e} max_geo_cpu={max_geo_cpu:.8e}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn hq30uq4_group128_geo_matches_cpu_small_and_above_wrap() {
        let context = match crate::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skip: MetalContext::new failed: {err}");
                return;
            }
        };

        const SMALL_ROWS: usize = 16;
        const SMALL_COLS: usize = 256;
        let fill: Vec<usize> = (0..SMALL_ROWS).collect();
        let (scales, codes) = patterned_hq30uq4_planes(SMALL_ROWS, SMALL_COLS, 128, &fill);
        let x = ramp_x(SMALL_COLS);
        let (name, grid, tg) =
            qwen38_uniform_q4_geo_tpr64_launch(128, SMALL_ROWS as u32, SMALL_COLS as u32)
                .expect("g128 small bind");
        assert_eq!(name, QWEN38_Q4_GROUP128_MATVEC_KERNEL);
        let geo = dispatch_hq30uq4_geo(
            &context,
            name,
            grid,
            tg,
            &codes,
            &scales,
            &x,
            SMALL_ROWS as u32,
            SMALL_COLS as u32,
            (SMALL_COLS / 128) as u32,
        );
        let mut small_max = 0.0f32;
        for row in 0..SMALL_ROWS {
            let cpu = hq30uq4_cpu_row(&codes, &scales, 128, row, SMALL_COLS, &x);
            let d = (geo[row] - cpu).abs();
            small_max = small_max.max(d);
            eprintln!(
                "G128_SMALL row={row} cpu={cpu:.8e} geo={:.8e} abs_d={d:.3e}",
                geo[row]
            );
            assert_eq!(d, 0.0, "g128 small row {row} vs CPU");
        }
        eprintln!("G128_SMALL_SUMMARY max_abs={small_max:.8e}");

        // element*4 wraps at row 209715 when K=5120. Allocate that height so
        // rgb = row * (5120/128) is the production address, not a small-tensor
        // stand-in. Only fill the wrap neighborhood and the last rows.
        const TALL_ROWS: usize = 209720;
        const TALL_COLS: usize = 5120;
        const WRAP_ROW: usize = 209715;
        let probe = [
            0usize,
            1,
            WRAP_ROW - 1,
            WRAP_ROW,
            WRAP_ROW + 1,
            TALL_ROWS - 2,
            TALL_ROWS - 1,
        ];
        let (scales, codes) = patterned_hq30uq4_planes(TALL_ROWS, TALL_COLS, 128, &probe);
        let x = ramp_x(TALL_COLS);
        let (name, grid, tg) =
            qwen38_uniform_q4_geo_tpr64_launch(128, TALL_ROWS as u32, TALL_COLS as u32)
                .expect("g128 tall bind");
        let geo = dispatch_hq30uq4_geo(
            &context,
            name,
            grid,
            tg,
            &codes,
            &scales,
            &x,
            TALL_ROWS as u32,
            TALL_COLS as u32,
            (TALL_COLS / 128) as u32,
        );
        eprintln!(
            "G128_ABOVE_WRAP header: row cpu geo abs_d codes={} scales={}",
            codes.len(),
            scales.len() * 2
        );
        let mut wrap_max = 0.0f32;
        for &row in &probe {
            let cpu = hq30uq4_cpu_row(&codes, &scales, 128, row, TALL_COLS, &x);
            let d = (geo[row] - cpu).abs();
            wrap_max = wrap_max.max(d);
            eprintln!(
                "G128_ABOVE_WRAP row={row} cpu={cpu:.8e} geo={:.8e} abs_d={d:.3e}",
                geo[row]
            );
            assert_eq!(d, 0.0, "g128 row {row} vs CPU above wrap");
        }
        eprintln!(
            "G128_ABOVE_WRAP_SUMMARY wrap_row={WRAP_ROW} max_abs={wrap_max:.8e} rows={TALL_ROWS} cols={TALL_COLS}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn g0_lm_head_geo_matches_cpu_above_uint32_wrap() {
        let root = campaign_qwen38("uniform-q4-v1");
        if !root.join("manifest.json").is_file() {
            eprintln!("skip: uniform-q4-v1 not on this host");
            return;
        }
        let context = match crate::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skip: MetalContext::new failed: {err}");
                return;
            }
        };
        let (_, rows) = load_qwen38_manifest(&root).unwrap();
        let row = rows
            .iter()
            .find(|r| r.name.ends_with("lm_head.weight"))
            .expect("G0 lm_head");
        let payload = std::fs::read(root.join("tensors").join(&row.artifact)).unwrap();
        let header = parse_uniform_q4_header(&payload).unwrap();
        assert_eq!(header.group_size, 64, "G0 lm_head must stay group 64");
        assert_eq!(header.shape.as_slice(), &[248320, 5120]);
        let scales: Vec<u16> = payload[header.scale_offset..header.sign_offset]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let codes = &payload[header.sign_offset..header.payload_bytes];
        let rows_n = header.shape[0];
        let cols = header.shape[1];
        let x = ramp_x(cols);
        let (name, grid, tg) =
            qwen38_uniform_q4_geo_tpr64_launch(64, rows_n as u32, cols as u32).expect("g0 bind");
        assert_eq!(name, qwen38_q4_kernel(QWEN38_Q4_MATVEC_KERNEL));
        let geo = dispatch_hq30uq4_geo(
            &context,
            name,
            grid,
            tg,
            codes,
            &scales,
            &x,
            rows_n as u32,
            cols as u32,
            (cols / 64) as u32,
        );
        let serial = dispatch_hq30uq4_geo(
            &context,
            "qwen_uniform_q4_group64_matvec",
            (rows_n as u32, 1, 1),
            (256, 1, 1),
            codes,
            &scales,
            &x,
            rows_n as u32,
            cols as u32,
            (cols / 64) as u32,
        );
        const WRAP: usize = 209715;
        let probe = [0usize, WRAP - 1, WRAP, WRAP + 1, rows_n - 2, rows_n - 1];
        eprintln!("G0_LM_HEAD header: row cpu serial geo d_serial d_cpu");
        let mut max_serial = 0.0f32;
        let mut max_cpu = 0.0f32;
        for &r in &probe {
            let cpu = hq30uq4_cpu_row(codes, &scales, 64, r, cols, &x);
            let d_serial = (geo[r] - serial[r]).abs();
            let d_cpu = (geo[r] - cpu).abs();
            max_serial = max_serial.max(d_serial);
            max_cpu = max_cpu.max(d_cpu);
            eprintln!(
                "G0_LM_HEAD row={r} cpu={cpu:.8e} serial={:.8e} geo={:.8e} d_serial={d_serial:.3e} d_cpu={d_cpu:.3e}",
                serial[r], geo[r]
            );
            assert_eq!(d_serial, 0.0, "G0 lm_head row {r} geo vs serial");
            assert_eq!(d_cpu, 0.0, "G0 lm_head row {r} geo vs CPU");
        }
        let mut n_serial_mismatch = 0usize;
        for i in 0..rows_n {
            if geo[i] != serial[i] {
                n_serial_mismatch += 1;
            }
        }
        eprintln!(
            "G0_LM_HEAD_SUMMARY rows={rows_n} cols={cols} wrap={WRAP} max_d_serial={max_serial:.8e} max_d_cpu={max_cpu:.8e} n_geo_ne_serial={n_serial_mismatch}"
        );
        assert_eq!(n_serial_mismatch, 0, "G0 geo must be bit-identical to serial");
    }

    #[test]
    fn group128_kernel_addressing_matches_cpu_at_wrap_row() {
        // CPU stand-in for the sibling's (row, group) ulong walk at the
        // historical element*4 wrap row. Does not allocate the 537 MiB plane.
        const COLS: usize = 5120;
        const GROUP: usize = 128;
        const GPR: usize = COLS / GROUP;
        const WRAP: usize = 209715;
        let x = ramp_x(COLS);
        let one = half::f16::from_f32(1.0).to_bits();
        let probe = [WRAP - 1, WRAP, WRAP + 1, 248319];
        eprintln!(
            "G128_ADDR_ORACLE header: row cpu kernel abs_d rgb0 u32_rgb0_wraps"
        );
        for &row in &probe {
            let mut codes = vec![0u8; GPR * 64];
            let mut scales = vec![0u16; GPR];
            for group in 0..GPR {
                scales[group] = one;
                for local in 0..GROUP {
                    let nibble = ((row + group + local) & 0x0f) as u8;
                    let byte = group * 64 + local / 2;
                    if local & 1 == 0 {
                        codes[byte] |= nibble;
                    } else {
                        codes[byte] |= nibble << 4;
                    }
                }
            }
            let cpu = hq30uq4_cpu_row(&codes, &scales, GROUP, 0, COLS, &x);
            // Kernel walk: 64 lanes, col = lane*8 + 512k, 8-wide unpack.
            let mut kernel = 0.0f32;
            for lane in 0..64u32 {
                let mut col = lane * 8;
                while (col as usize) < COLS {
                    let group = (col as usize) / GROUP;
                    let local = (col as usize) % GROUP;
                    let rgb0 = (row as u64) * (GPR as u64);
                    let rgb = rgb0 + group as u64;
                    let code_off = rgb * 64 + (local as u64 / 2);
                    // Plane is stored as one row; rgb's group index is `group`.
                    let local_off = (code_off - rgb0 * 64) as usize;
                    let packed = u32::from_le_bytes(codes[local_off..local_off + 4].try_into().unwrap());
                    let scale = half::f16::from_bits(scales[group]).to_f32();
                    for i in 0..4u32 {
                        let byte = (packed >> (8 * i)) & 0xff;
                        let c0 = col + 2 * i;
                        let c1 = c0 + 1;
                        kernel += (i32::from((byte & 0x0f) as u8) - 8) as f32 * scale * x[c0 as usize];
                        kernel += (i32::from((byte >> 4) as u8) - 8) as f32 * scale * x[c1 as usize];
                    }
                    col += 512;
                }
            }
            let d = (kernel - cpu).abs();
            let rgb0 = (row as u64) * (GPR as u64);
            let u32_wraps = (row as u32).wrapping_mul(GPR as u32) as u64 != rgb0;
            eprintln!(
                "G128_ADDR_ORACLE row={row} cpu={cpu:.8e} kernel={kernel:.8e} abs_d={d:.3e} rgb0={rgb0} u32_rgb0_wraps={u32_wraps}"
            );
            assert!(!u32_wraps, "row*{GPR} must not wrap u32 at row {row}");
            assert_eq!(d, 0.0, "kernel walk vs CPU at row {row}");
        }
    }

    #[test]
    fn group128_code_offset_is_u64_because_u32_wraps() {
        // New indexing: byte = rgb * 64. In uint32 that product wraps at
        // rgb >= 2^26 = 67_108_864. The sibling forms it in ulong.
        let wrap_rgb = 1u64 << 26;
        let u32_off = (wrap_rgb as u32).wrapping_mul(64);
        let u64_off = wrap_rgb * 64;
        assert_eq!(u32_off, 0, "uint32 rgb*64 must wrap at 2^26");
        assert_eq!(u64_off, 1u64 << 32);
        let lm_rgb_max = 248320u64 * (5120 / 128) - 1;
        assert!(lm_rgb_max < wrap_rgb, "lm_head g128 rgb_max={lm_rgb_max}");
        assert!((lm_rgb_max as u32 as u64) * 64 == lm_rgb_max * 64);
        let first_overflow_row = ((1u64 << 32) / 4) / 5120;
        assert_eq!(first_overflow_row, 209715);
        let rgb_at_wrap_row = 209715u64 * (5120 / 128);
        assert!(rgb_at_wrap_row * 64 < (1u64 << 32));
        let src = crate::metal::SHADER_QWEN_UNIFORM_Q4;
        let g128 = src
            .split("kernel void qwen_uniform_q4_group128_matvec_geo_tpr64_tg128(")
            .nth(1)
            .expect("g128 kernel");
        assert!(g128.contains("const ulong rgb0 = (ulong)row * (ulong)groups_per_row;"));
        assert!(g128.contains("rgb * (ulong)QWEN_UNIFORM_Q4_CODE_BYTES_PER_GROUP_128"));
        assert!(!g128.contains("element * bits"));
    }

    #[test]
    fn uint32_element_times_bits_wraps_on_lm_head_for_on_disk_widths() {
        const ROWS: u64 = 248320;
        const COLS: u64 = 5120;
        let elements = ROWS * COLS;
        eprintln!(
            "WRAP_TABLE header: bits wrap_el first_row lm_head_elements reaches"
        );
        for bits in [3u32, 4, 5, 6, 7, 8] {
            let wrap_el = (1u64 << 32) / u64::from(bits);
            let first_row = wrap_el / COLS;
            let reaches = elements >= wrap_el;
            eprintln!("WRAP bits={bits} wrap_el={wrap_el} first_row={first_row} elements={elements} reaches={reaches}");
            if bits == 3 {
                assert!(!reaches, "bits=3 must not wrap on this lm_head");
            } else {
                assert!(reaches, "bits={bits} must wrap on this lm_head");
            }
        }
        assert_eq!(uint32_first_overflow_row(4, 5120), 209715);
        assert_eq!(uint32_first_overflow_row(7, 5120), 119837);
        assert_eq!(uint32_first_overflow_row(8, 5120), 104857);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn incumbent_extract_matches_cpu_oracle_above_uint32_overflow() {
        // The production geo_tpr64 bind only covers bits 3/4. bits 5–8
        // still dispatch the incumbent, which used `uint bit0 = element * bits`.
        // Existing parity compared incumbent vs incumbent on tensors below
        // the wrap, so both sides were identically correct. This test
        // compares the overflowing incumbent against the CPU usize oracle
        // on embed and lm_head at their real 248320×5120 size, at every
        // HGRAVU01 bit width those tensors actually use on disk (4, 7, 8).
        let cases = [
            ("mixed-q3mlp-v1", 4u32),
            ("mixed-floor-q7-v1", 7u32),
            ("mixed-floor-q8-v1", 8u32),
        ];
        let tensors = [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
        ];
        let context = match crate::metal::MetalContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                eprintln!("skip: MetalContext::new failed: {err}");
                return;
            }
        };
        let mut saw = 0usize;
        let mut failures: Vec<String> = Vec::new();
        eprintln!(
            "ABOVE_OVERFLOW header: artifact tensor bits kernel rows cols first_ov row cpu gpu abs_d"
        );
        for (artifact, expect_bits) in cases {
            let root = campaign_qwen38(artifact);
            if !root.join(QWEN38_MIXED_CATALOG_NAME).is_file() {
                eprintln!("skip artifact: {artifact} catalog missing");
                continue;
            }
            let rows = parse_qwen38_mixed_catalog(&root).unwrap();
            for name in tensors {
                let row = rows
                    .iter()
                    .find(|r| r.name == name)
                    .unwrap_or_else(|| panic!("{artifact} missing {name}"));
                let payload = read_catalog_payload(row).unwrap();
                let layout = mixed_gpu_layout(row.codec, &payload)
                    .unwrap_or_else(|e| panic!("{artifact} {name}: {e}"));
                let MixedGpuKind::Uniform(factor) = layout.kind else {
                    panic!("{artifact} {name} is not HGRAVU01 Uniform");
                };
                assert_eq!(
                    factor.bits, expect_bits,
                    "{artifact} {name} bits"
                );
                assert_eq!(factor.rows, 248320, "{artifact} {name} rows");
                assert_eq!(factor.cols, 5120, "{artifact} {name} cols");
                let kernel = overflowing_incumbent_kernel(factor.bits);
                assert!(
                    !kernel.contains("geo_tpr64") && !kernel.contains("uniform8"),
                    "must exercise the overflowing extract, got {kernel}"
                );
                let first_ov = uint32_first_overflow_row(factor.bits, factor.cols);
                assert!(
                    first_ov < factor.rows as usize,
                    "{artifact} {name} bits={} does not reach wrap",
                    factor.bits
                );
                let codes = context
                    .new_buffer_with_bytes_checked(
                        &payload[factor.code_off..factor.code_off + factor.code_bytes],
                    )
                    .unwrap();
                let scales = context
                    .new_buffer_with_bytes_checked(
                        &payload[factor.scale_off..factor.scale_off + factor.scale_bytes],
                    )
                    .unwrap();
                let mut x = vec![0.0f32; factor.cols as usize];
                for (i, slot) in x.iter_mut().enumerate() {
                    *slot = (i % 17) as f32 * 0.125 - 1.0;
                }
                let input = context
                    .new_buffer_with_bytes_checked(bytemuck::cast_slice(&x))
                    .unwrap();
                let out = context
                    .new_buffer_checked(factor.rows as usize * 4)
                    .unwrap();
                let inc_grid = (
                    factor.rows.div_ceil(8).saturating_mul(256).max(256),
                    1,
                    1,
                );
                let mut tcb = crate::metal::TokenCommandBuffer::new(&context);
                tcb.dispatch_threads(kernel, inc_grid, (256, 1, 1), |enc| {
                    enc.set_buffer(0, Some(&codes), 0);
                    enc.set_buffer(1, Some(&scales), 0);
                    enc.set_buffer(2, Some(&input), 0);
                    enc.set_buffer(3, Some(&out), 0);
                    for (index, value) in [
                        (4u64, factor.rows),
                        (5, factor.cols),
                        (6, factor.group_size),
                        (7, factor.bits),
                        (8, factor.bound),
                    ] {
                        enc.set_bytes(index, 4, &value as *const u32 as *const _);
                    }
                })
                .unwrap();
                tcb.commit_and_wait().unwrap();
                let n = factor.rows as usize;
                let gpu = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, n) };
                let packed = packed_from_uniform_payload(&factor, &payload);
                drop(payload);
                let mut probe = vec![
                    first_ov.saturating_sub(1),
                    first_ov,
                    first_ov + 1,
                    n - 2,
                    n - 1,
                ];
                for r in 248044..=248076 {
                    if r < n {
                        probe.push(r);
                    }
                }
                probe.sort_unstable();
                probe.dedup();
                probe.retain(|&r| r < n);
                for r in probe {
                    let mut cpu = 0.0f32;
                    for c in 0..packed.cols {
                        cpu += uniform_factor_value(&packed, r, c) * x[c];
                    }
                    let g = gpu[r];
                    let d = (g - cpu).abs();
                    eprintln!(
                        "ABOVE_OVERFLOW {artifact} {name} bits={} {kernel} {}x{} first_ov={first_ov} row={r} cpu={cpu:.8e} gpu={g:.8e} abs_d={d:.3e}",
                        factor.bits, factor.rows, factor.cols
                    );
                    if r < first_ov {
                        if d > 1.0e-4 {
                            failures.push(format!(
                                "{artifact} {name} bits={} row={r} BELOW wrap should match: cpu={cpu} gpu={g} d={d}",
                                factor.bits
                            ));
                        }
                    } else if d > 1.0e-4 {
                        failures.push(format!(
                            "{artifact} {name} bits={} row={r} ABOVE wrap: cpu={cpu:.8e} gpu={g:.8e} d={d:.3e}",
                            factor.bits
                        ));
                    }
                }
                saw += 1;
            }
        }
        assert!(
            saw == 6,
            "expected 3 artifacts × 2 tensors, saw {saw} (missing on-disk HGRAVU01 embed/lm_head)"
        );
        assert!(
            failures.is_empty(),
            "incumbent extract != CPU usize oracle above uint32 wrap ({}):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

#[cfg(test)]
mod complete_wall_identity_tests {
    use super::{Qwen38CompleteToken, Qwen38StepWall};

    #[test]
    fn step_named_sum_plus_residual_equals_wall() {
        let step = Qwen38StepWall {
            wall_ns: 34_000_000,
            encode_ns: 400_000,
            submit_ns: 20_000,
            wait_ns: 33_500_000,
            gpu_ns: Some(33_100_000),
            gpu_start_s: None,
            gpu_end_s: None,
            gpu_start_ns: None,
            gpu_end_ns: None,
            allocation_ns: 0,
            encoder_count: 900,
            commit_epilogue_ns: 30_000,
            sample_readback_ns: 2_000,
            state_update_ns: 1_000,
            tcb_encode_ns: 0,
            dispatches: 900,
            command_buffers: 1,
            active_weight_bytes: 0,
        };
        assert_eq!(
            step.named_sum_ns() as i64 + step.residual_ns(),
            step.wall_ns as i64
        );
        assert_eq!(step.wait_minus_gpu_ns(), Some(400_000));
    }

    #[test]
    fn complete_token_names_tokenizer_and_bookkeeping() {
        let token = Qwen38CompleteToken {
            role: "decode".into(),
            step_index: 12,
            token_in: 1,
            token_out: 2,
            step: Qwen38StepWall {
                wall_ns: 33_953_000,
                encode_ns: 400_000,
                submit_ns: 20_000,
                wait_ns: 33_500_000,
                gpu_ns: Some(33_100_000),
                gpu_start_s: None,
                gpu_end_s: None,
                gpu_start_ns: None,
                gpu_end_ns: None,
                allocation_ns: 0,
                encoder_count: 900,
                commit_epilogue_ns: 30_000,
                sample_readback_ns: 2_000,
                state_update_ns: 1_000,
                tcb_encode_ns: 0,
                dispatches: 900,
                command_buffers: 1,
                active_weight_bytes: 0,
            },
            tokenizer_decode_ns: 8_000,
            bookkeeping_ns: 1_000,
            complete_wall_ns: 33_970_000,
        };
        assert_eq!(
            token.named_sum_ns() as i64 + token.residual_ns(),
            token.complete_wall_ns as i64
        );
        assert_eq!(token.wall_minus_gpu_ns(), Some(870_000));
    }
}

#[cfg(test)]
mod workspace_bytes_tests {
    use super::qwen38_workspace_bytes;

    #[test]
    fn rejects_zero_seq() {
        assert!(qwen38_workspace_bytes(0).is_err());
    }

    #[test]
    fn kv_is_the_seq_len_term() {
        let a = qwen38_workspace_bytes(2048).unwrap();
        let b = qwen38_workspace_bytes(4096).unwrap();
        assert_eq!(a.activation_bytes, b.activation_bytes);
        assert_eq!(a.deltanet_state_bytes, b.deltanet_state_bytes);
        assert_eq!(b.gqa_kv_bytes, a.gqa_kv_bytes * 2);
        assert_eq!(
            b.total_bytes - a.total_bytes,
            b.gqa_kv_bytes - a.gqa_kv_bytes
        );
    }

    #[test]
    fn seq2048_is_hundreds_of_mb_not_weight_sized() {
        let bytes = qwen38_workspace_bytes(2048).unwrap();
        assert!(
            bytes.total_bytes > 200 * 1024 * 1024,
            "workspace {}",
            bytes.total_bytes
        );
        assert!(
            bytes.total_bytes < 2 * 1024 * 1024 * 1024,
            "workspace {} must stay far below the 8.5 GB artifact",
            bytes.total_bytes
        );
        assert!(bytes.deltanet_state_bytes > bytes.gqa_kv_bytes / 4);
    }
}
