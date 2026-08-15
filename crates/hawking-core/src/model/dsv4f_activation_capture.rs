//! DSV4F activation-X capture writer.
//!
//! Memory-bounded, deterministic, per-(layer, expert) first-N retention that
//! emits real float32-LE X rows in the same on-disk shape Q80 already uses, so
//! doctor6's `collect_expert_activations` can read the run unchanged.
//!
//! This module does **not** run a 43-layer source forward. A later scheduled
//! job supplies real layer activations; tests and the example prove the writer
//! on a reduced configuration. Layers >= 2 of a real source still need the
//! Metal sparse-attention / indexer graph (out of scope here).
//!
//! Organ list is the sealed `activation_x_capture` catalog from
//! `receipts/DSV4F_TENSOR_SCHEDULE.json`. It is not re-derived.

use crate::{Error, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Sealed base-body geometry (DSV4F_TENSOR_SCHEDULE / runtime spine).
pub const DSV4F_CAPTURE_LAYERS: usize = 43;
pub const DSV4F_CAPTURE_HIDDEN: usize = 4096;
pub const DSV4F_CAPTURE_ROUTED_EXPERTS: usize = 256;
pub const DSV4F_CAPTURE_SHARED_EXPERTS: usize = 1;
pub const DSV4F_CAPTURE_TOP_K: usize = 6;
pub const DSV4F_CAPTURE_MOE_INTERMEDIATE: usize = 2048;
pub const DSV4F_CAPTURE_Q_LORA: usize = 1024;
pub const DSV4F_CAPTURE_O_LORA: usize = 8192;
pub const DSV4F_CAPTURE_HC_FLAT: usize = 16_384;
pub const DSV4F_CAPTURE_HASH_LAYERS: usize = 3;
pub const DSV4F_CAPTURE_REQUIRED_ORGANS: usize = 18;

/// Default retained router-input rows per (layer, expert) under first-N.
pub const DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT: usize = 64;
/// Default n_fit "enough rows" threshold reported in coverage stats.
pub const DEFAULT_ROW_THRESHOLD: usize = 16;

pub const RESULT_SCHEMA: &str = "hawking.dsv4f.activation_x_capture_result.v1";
pub const CAPTURE_PROTOCOL_REVISION: &str = "dsv4f-activation-x-capture-per-expert-first-n-v1";

/// Official inference compress_ratios: 43 base + 1 MTP trailer.
pub const COMPRESS_RATIOS: [u32; 44] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
];

fn model_err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

/// Distinct X matrices that size the capture (sealed schedule).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum XId {
    HPostAttnNorm,
    QLoraQr,
    AttnOutGrouped,
    OLora,
    HPostFfnNorm,
    SwigluHiddenRouted,
    SwigluHiddenShared,
    HcFlatPreAttn,
    HcFlatPreFfn,
    HFinal,
}

impl XId {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HPostAttnNorm => "h_post_attn_norm",
            Self::QLoraQr => "q_lora_qr",
            Self::AttnOutGrouped => "attn_out_grouped",
            Self::OLora => "o_lora",
            Self::HPostFfnNorm => "h_post_ffn_norm",
            Self::SwigluHiddenRouted => "swiglu_hidden_routed",
            Self::SwigluHiddenShared => "swiglu_hidden_shared",
            Self::HcFlatPreAttn => "hc_flat_pre_attn",
            Self::HcFlatPreFfn => "hc_flat_pre_ffn",
            Self::HFinal => "h_final",
        }
    }

    pub fn sealed_dim(self) -> usize {
        match self {
            Self::HPostAttnNorm | Self::AttnOutGrouped | Self::HPostFfnNorm | Self::HFinal => {
                DSV4F_CAPTURE_HIDDEN
            }
            Self::QLoraQr => DSV4F_CAPTURE_Q_LORA,
            Self::OLora => DSV4F_CAPTURE_O_LORA,
            Self::SwigluHiddenRouted | Self::SwigluHiddenShared => DSV4F_CAPTURE_MOE_INTERMEDIATE,
            Self::HcFlatPreAttn | Self::HcFlatPreFfn => DSV4F_CAPTURE_HC_FLAT,
        }
    }

    pub fn is_optional(self) -> bool {
        matches!(
            self,
            Self::HcFlatPreAttn | Self::HcFlatPreFfn | Self::HFinal
        )
    }

    pub fn all() -> &'static [XId] {
        &[
            Self::HPostAttnNorm,
            Self::QLoraQr,
            Self::AttnOutGrouped,
            Self::OLora,
            Self::HPostFfnNorm,
            Self::SwigluHiddenRouted,
            Self::SwigluHiddenShared,
            Self::HcFlatPreAttn,
            Self::HcFlatPreFfn,
            Self::HFinal,
        ]
    }
}

/// Where a sealed organ is present.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrganLayerScope {
    AllBase,
    NonSliding,
    Ratio4Only,
    AfterLastBase,
    MtpOnly,
}

/// One organ from the sealed `activation_x_capture.organs` list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CaptureOrgan {
    pub name: &'static str,
    pub input_dim: usize,
    pub x_id: XId,
    pub required_for_1_5_complete: bool,
    pub scope: OrganLayerScope,
}

/// Sealed required + optional organs. Do not re-derive; this is the schedule.
pub const CAPTURE_ORGANS: &[CaptureOrgan] = &[
    CaptureOrgan {
        name: "routed_expert.w1",
        input_dim: 4096,
        x_id: XId::HPostFfnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "routed_expert.w3",
        input_dim: 4096,
        x_id: XId::HPostFfnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "routed_expert.w2",
        input_dim: 2048,
        x_id: XId::SwigluHiddenRouted,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "shared_expert.w1",
        input_dim: 4096,
        x_id: XId::HPostFfnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "shared_expert.w3",
        input_dim: 4096,
        x_id: XId::HPostFfnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "shared_expert.w2",
        input_dim: 2048,
        x_id: XId::SwigluHiddenShared,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "mla.wq_a",
        input_dim: 4096,
        x_id: XId::HPostAttnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "mla.wq_b",
        input_dim: 1024,
        x_id: XId::QLoraQr,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "mla.wkv",
        input_dim: 4096,
        x_id: XId::HPostAttnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "mla.wo_a",
        input_dim: 4096,
        x_id: XId::AttnOutGrouped,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "mla.wo_b",
        input_dim: 8192,
        x_id: XId::OLora,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "indexer.wq_b",
        input_dim: 1024,
        x_id: XId::QLoraQr,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::Ratio4Only,
    },
    CaptureOrgan {
        name: "indexer.weights_proj",
        input_dim: 4096,
        x_id: XId::HPostAttnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::Ratio4Only,
    },
    CaptureOrgan {
        name: "compressor.wkv",
        input_dim: 4096,
        x_id: XId::HPostAttnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::NonSliding,
    },
    CaptureOrgan {
        name: "compressor.wgate",
        input_dim: 4096,
        x_id: XId::HPostAttnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::NonSliding,
    },
    CaptureOrgan {
        name: "indexer.compressor.wkv",
        input_dim: 4096,
        x_id: XId::HPostAttnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::Ratio4Only,
    },
    CaptureOrgan {
        name: "indexer.compressor.wgate",
        input_dim: 4096,
        x_id: XId::HPostAttnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::Ratio4Only,
    },
    CaptureOrgan {
        name: "router_gate.weight",
        input_dim: 4096,
        x_id: XId::HPostFfnNorm,
        required_for_1_5_complete: true,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "mhc.hc_attn_fn",
        input_dim: 16384,
        x_id: XId::HcFlatPreAttn,
        required_for_1_5_complete: false,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "mhc.hc_ffn_fn",
        input_dim: 16384,
        x_id: XId::HcFlatPreFfn,
        required_for_1_5_complete: false,
        scope: OrganLayerScope::AllBase,
    },
    CaptureOrgan {
        name: "lm_head",
        input_dim: 4096,
        x_id: XId::HFinal,
        required_for_1_5_complete: false,
        scope: OrganLayerScope::AfterLastBase,
    },
    CaptureOrgan {
        name: "mtp.e_proj",
        input_dim: 4096,
        x_id: XId::HPostFfnNorm,
        required_for_1_5_complete: false,
        scope: OrganLayerScope::MtpOnly,
    },
    CaptureOrgan {
        name: "mtp.h_proj",
        input_dim: 4096,
        x_id: XId::HPostFfnNorm,
        required_for_1_5_complete: false,
        scope: OrganLayerScope::MtpOnly,
    },
];

pub fn required_organs() -> impl Iterator<Item = &'static CaptureOrgan> {
    CAPTURE_ORGANS
        .iter()
        .filter(|o| o.required_for_1_5_complete)
}

pub fn compress_ratio(layer: usize) -> Option<u32> {
    COMPRESS_RATIOS.get(layer).copied()
}

pub fn organ_applies_to_layer(organ: &CaptureOrgan, layer: usize) -> bool {
    match organ.scope {
        OrganLayerScope::AllBase => layer < DSV4F_CAPTURE_LAYERS,
        OrganLayerScope::NonSliding => {
            layer < DSV4F_CAPTURE_LAYERS && compress_ratio(layer) != Some(0)
        }
        OrganLayerScope::Ratio4Only => {
            layer < DSV4F_CAPTURE_LAYERS && compress_ratio(layer) == Some(4)
        }
        OrganLayerScope::AfterLastBase => layer + 1 == DSV4F_CAPTURE_LAYERS,
        OrganLayerScope::MtpOnly => false,
    }
}

/// `true` iff at least one required organ that feeds on `x` is present at `layer`.
pub fn required_x_applies_to_layer(x: XId, layer: usize) -> bool {
    CAPTURE_ORGANS
        .iter()
        .any(|o| o.required_for_1_5_complete && o.x_id == x && organ_applies_to_layer(o, layer))
}

/// Capture geometry. Defaults to the sealed model; tests may shrink it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaptureGeometry {
    pub layers: usize,
    pub hidden: usize,
    pub routed_experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub q_lora: usize,
    pub o_lora: usize,
    pub hc_flat: usize,
}

impl CaptureGeometry {
    pub fn sealed() -> Self {
        Self {
            layers: DSV4F_CAPTURE_LAYERS,
            hidden: DSV4F_CAPTURE_HIDDEN,
            routed_experts: DSV4F_CAPTURE_ROUTED_EXPERTS,
            top_k: DSV4F_CAPTURE_TOP_K,
            moe_intermediate: DSV4F_CAPTURE_MOE_INTERMEDIATE,
            q_lora: DSV4F_CAPTURE_Q_LORA,
            o_lora: DSV4F_CAPTURE_O_LORA,
            hc_flat: DSV4F_CAPTURE_HC_FLAT,
        }
    }

    /// Small fixture for unit tests. Expert table stays large enough to prove
    /// per-expert fairness; row width is tiny so disk/RAM stay cheap.
    pub fn test_fixture() -> Self {
        Self {
            layers: 4,
            hidden: 16,
            routed_experts: 32,
            top_k: DSV4F_CAPTURE_TOP_K,
            moe_intermediate: 8,
            q_lora: 8,
            o_lora: 16,
            hc_flat: 32,
        }
    }

    pub fn dim_of(&self, x: XId) -> usize {
        match x {
            XId::HPostAttnNorm | XId::AttnOutGrouped | XId::HPostFfnNorm | XId::HFinal => {
                self.hidden
            }
            XId::QLoraQr => self.q_lora,
            XId::OLora => self.o_lora,
            XId::SwigluHiddenRouted | XId::SwigluHiddenShared => self.moe_intermediate,
            XId::HcFlatPreAttn | XId::HcFlatPreFfn => self.hc_flat,
        }
    }
}

/// Which X matrices to retain. Router-input (`h_post_ffn_norm`) is always on:
/// that is the doctor6 `router_input_hidden_f32le` row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CaptureSet {
    pub h_post_attn_norm: bool,
    pub q_lora_qr: bool,
    pub attn_out_grouped: bool,
    pub o_lora: bool,
    pub swiglu_hidden_routed: bool,
    pub swiglu_hidden_shared: bool,
    pub optional_mhc: bool,
    pub optional_lm_head: bool,
}

impl CaptureSet {
    /// Every required organ in the sealed catalog.
    pub fn required() -> Self {
        Self {
            h_post_attn_norm: true,
            q_lora_qr: true,
            attn_out_grouped: true,
            o_lora: true,
            swiglu_hidden_routed: true,
            swiglu_hidden_shared: true,
            optional_mhc: false,
            optional_lm_head: false,
        }
    }

    /// Doctor6 collector surface only (routed-expert w1/w3 X).
    pub fn doctor6_only() -> Self {
        Self {
            h_post_attn_norm: false,
            q_lora_qr: false,
            attn_out_grouped: false,
            o_lora: false,
            swiglu_hidden_routed: false,
            swiglu_hidden_shared: false,
            optional_mhc: false,
            optional_lm_head: false,
        }
    }

    pub fn includes(&self, x: XId) -> bool {
        match x {
            XId::HPostFfnNorm => true,
            XId::HPostAttnNorm => self.h_post_attn_norm,
            XId::QLoraQr => self.q_lora_qr,
            XId::AttnOutGrouped => self.attn_out_grouped,
            XId::OLora => self.o_lora,
            XId::SwigluHiddenRouted => self.swiglu_hidden_routed,
            XId::SwigluHiddenShared => self.swiglu_hidden_shared,
            XId::HcFlatPreAttn | XId::HcFlatPreFfn => self.optional_mhc,
            XId::HFinal => self.optional_lm_head,
        }
    }
}

/// Deterministic per-expert first-N retention for one token.
///
/// Walk tokens in global order. For each top-k expert that still has fewer than
/// `max_per_expert` retained members, credit this token to that expert. Retain
/// the token's router-input hidden if any expert still needed a slot.
///
/// This is deliberately not a random reservoir: the same corpus + same N yields
/// byte-identical retention across runs. Route membership is independent and
/// stays complete for every token.
pub fn credit_expert_first_n_retention(
    expert_retained: &mut [usize],
    selected_expert_ids: &[u32],
    max_per_expert: usize,
) -> bool {
    if max_per_expert == 0 || selected_expert_ids.is_empty() {
        return false;
    }
    let mut retain = false;
    for &expert in selected_expert_ids {
        let e = expert as usize;
        if e >= expert_retained.len() {
            continue;
        }
        if expert_retained[e] < max_per_expert {
            expert_retained[e] += 1;
            retain = true;
        }
    }
    retain
}

/// Extra (non-router-input) X payloads for one token at one layer.
///
/// Emptied on per-layer flush. `h_post_ffn_norm` lives on
/// [`LayerTokenCapture::router_input_hidden`] so the doctor6 field has a
/// single owner.
#[derive(Clone, Debug, Default)]
pub struct TokenXPayloads {
    pub h_post_attn_norm: Vec<f32>,
    pub q_lora_qr: Vec<f32>,
    pub attn_out_grouped: Vec<f32>,
    pub o_lora: Vec<f32>,
    pub swiglu_hidden_shared: Vec<f32>,
    pub swiglu_hidden_routed: Vec<(u32, Vec<f32>)>,
    pub hc_flat_pre_attn: Vec<f32>,
    pub hc_flat_pre_ffn: Vec<f32>,
    pub h_final: Vec<f32>,
}

impl TokenXPayloads {
    pub fn f32_elements(&self) -> usize {
        self.h_post_attn_norm.len()
            + self.q_lora_qr.len()
            + self.attn_out_grouped.len()
            + self.o_lora.len()
            + self.swiglu_hidden_shared.len()
            + self
                .swiglu_hidden_routed
                .iter()
                .map(|(_, v)| v.len())
                .sum::<usize>()
            + self.hc_flat_pre_attn.len()
            + self.hc_flat_pre_ffn.len()
            + self.h_final.len()
    }

    pub fn clear(&mut self) {
        self.h_post_attn_norm = Vec::new();
        self.q_lora_qr = Vec::new();
        self.attn_out_grouped = Vec::new();
        self.o_lora = Vec::new();
        self.swiglu_hidden_shared = Vec::new();
        self.swiglu_hidden_routed = Vec::new();
        self.hc_flat_pre_attn = Vec::new();
        self.hc_flat_pre_ffn = Vec::new();
        self.h_final = Vec::new();
    }
}

/// Per-token capture surface for one layer.
///
/// After per-layer flush, [`Self::router_input_hidden`] and [`Self::extra_x`]
/// are empty even when [`Self::hidden_retained`] is true — the rows have been
/// written and freed. Route membership stays complete for every token.
#[derive(Clone, Debug)]
pub struct LayerTokenCapture {
    pub layer: usize,
    pub selected_expert_ids: Vec<u32>,
    pub normalized_route_weights: Vec<f32>,
    pub router_input_hidden: Vec<f32>,
    pub hidden_retained: bool,
    pub extra_x: TokenXPayloads,
}

/// All X rows for every token at one layer, packed `[token, dim]`.
#[derive(Clone, Debug, Default)]
pub struct LayerActivationBatch {
    pub h_post_ffn_norm: Vec<f32>,
    pub h_post_attn_norm: Option<Vec<f32>>,
    pub q_lora_qr: Option<Vec<f32>>,
    pub attn_out_grouped: Option<Vec<f32>>,
    pub o_lora: Option<Vec<f32>>,
    pub swiglu_hidden_shared: Option<Vec<f32>>,
    /// Per-token list of `(expert_id, swiglu_hidden)` for that token's route.
    pub swiglu_hidden_routed: Option<Vec<Vec<(u32, Vec<f32>)>>>,
    pub hc_flat_pre_attn: Option<Vec<f32>>,
    pub hc_flat_pre_ffn: Option<Vec<f32>>,
    pub h_final: Option<Vec<f32>>,
}

/// Bytes of retained hidden payloads currently sitting in `captures`.
///
/// After a per-layer flush this is the in-memory footprint of one layer (or
/// zero once that layer has been released). It must not grow with layer index.
pub fn resident_retained_hidden_bytes(captures: &[Vec<Vec<LayerTokenCapture>>]) -> usize {
    captures
        .iter()
        .flatten()
        .flatten()
        .map(|cap| {
            cap.router_input_hidden
                .len()
                .saturating_add(cap.extra_x.f32_elements())
                .saturating_mul(4)
        })
        .sum()
}

/// Drop retained hidden payloads for `layer_idx`. Route membership is left
/// intact; [`LayerTokenCapture::hidden_retained`] still records whether a row
/// was kept. Returns the number of `f32` elements freed.
pub fn release_layer_retained_hiddens(
    captures: &mut [Vec<Vec<LayerTokenCapture>>],
    layer_idx: usize,
) -> usize {
    let mut freed = 0usize;
    for probe in captures.iter_mut() {
        for token in probe.iter_mut() {
            for cap in token.iter_mut() {
                if cap.layer == layer_idx {
                    freed = freed.saturating_add(cap.router_input_hidden.len());
                    freed = freed.saturating_add(cap.extra_x.f32_elements());
                    cap.router_input_hidden = Vec::new();
                    cap.extra_x.clear();
                }
            }
        }
    }
    freed
}

/// Invoke `on_flush` (write this layer's retained rows) then drop those
/// payloads so they are not resident when the next layer loads.
pub fn flush_and_release_layer_hiddens<F>(
    captures: &mut [Vec<Vec<LayerTokenCapture>>],
    layer_idx: usize,
    on_flush: Option<&mut F>,
) -> Result<usize>
where
    F: FnMut(usize, &mut [Vec<Vec<LayerTokenCapture>>]) -> Result<()>,
{
    if let Some(cb) = on_flush {
        cb(layer_idx, captures)?;
    }
    Ok(release_layer_retained_hiddens(captures, layer_idx))
}

fn take_row(buf: &[f32], t: usize, width: usize, name: &str) -> Result<Vec<f32>> {
    let start = t
        .checked_mul(width)
        .ok_or_else(|| model_err("row start overflow"))?;
    let end = start
        .checked_add(width)
        .ok_or_else(|| model_err("row end overflow"))?;
    if buf.len() < end {
        return Err(model_err(format!(
            "{name}: token {t} needs elements [{start},{end}) but buffer has {}",
            buf.len()
        )));
    }
    Ok(buf[start..end].to_vec())
}

/// Apply per-expert first-N retention and append one layer of captures.
///
/// Expert-retained counters reset each call (each layer). Does not touch
/// residual streams. `routes` is consumed (moved into the capture rows).
pub fn append_retained_layer_captures(
    captures: &mut [Vec<Vec<LayerTokenCapture>>],
    token_index: &[(usize, usize)],
    routes: &mut [(Vec<u32>, Vec<f32>)],
    batch: &LayerActivationBatch,
    layer_idx: usize,
    geometry: &CaptureGeometry,
    set: &CaptureSet,
    max_hidden_tokens_per_expert: usize,
) -> Result<()> {
    let n = token_index.len();
    let h = geometry.hidden;
    if batch.h_post_ffn_norm.len() != n.saturating_mul(h) {
        return Err(model_err(format!(
            "h_post_ffn_norm length {} != {n}*{h}",
            batch.h_post_ffn_norm.len()
        )));
    }
    if routes.len() != n {
        return Err(model_err("routes/token_index length mismatch"));
    }
    if let Some(rows) = batch.swiglu_hidden_routed.as_ref() {
        if rows.len() != n {
            return Err(model_err(
                "swiglu_hidden_routed/token_index length mismatch",
            ));
        }
    }
    let mut expert_retained = vec![0usize; geometry.routed_experts];
    for (t, &(pi, pos)) in token_index.iter().enumerate() {
        if pi >= captures.len() || pos >= captures[pi].len() {
            return Err(model_err(format!(
                "token_index ({pi},{pos}) out of capture bounds"
            )));
        }
        let (ids, weights) = std::mem::take(&mut routes[t]);
        if ids.len() != geometry.top_k || weights.len() != geometry.top_k {
            return Err(model_err(format!(
                "token {t} route membership is {} ids / {} weights; expected top-{}",
                ids.len(),
                weights.len(),
                geometry.top_k
            )));
        }
        let retain = credit_expert_first_n_retention(
            &mut expert_retained,
            &ids,
            max_hidden_tokens_per_expert,
        );
        let mut extra = TokenXPayloads::default();
        let hidden = if retain {
            let row = take_row(&batch.h_post_ffn_norm, t, h, "h_post_ffn_norm")?;
            copy_extra_x(&mut extra, batch, t, layer_idx, geometry, set, &ids)?;
            row
        } else {
            Vec::new()
        };
        captures[pi][pos].push(LayerTokenCapture {
            layer: layer_idx,
            selected_expert_ids: ids,
            normalized_route_weights: weights,
            router_input_hidden: hidden,
            hidden_retained: retain,
            extra_x: extra,
        });
    }
    Ok(())
}

fn copy_extra_x(
    extra: &mut TokenXPayloads,
    batch: &LayerActivationBatch,
    t: usize,
    layer_idx: usize,
    geometry: &CaptureGeometry,
    set: &CaptureSet,
    selected: &[u32],
) -> Result<()> {
    let copy_dense = |src: &Option<Vec<f32>>, dest: &mut Vec<f32>, x: XId| -> Result<()> {
        if !set.includes(x) {
            return Ok(());
        }
        if x == XId::HFinal && layer_idx + 1 != DSV4F_CAPTURE_LAYERS {
            return Ok(());
        }
        if let Some(buf) = src.as_ref() {
            *dest = take_row(buf, t, geometry.dim_of(x), x.as_str())?;
        }
        Ok(())
    };
    copy_dense(
        &batch.h_post_attn_norm,
        &mut extra.h_post_attn_norm,
        XId::HPostAttnNorm,
    )?;
    copy_dense(&batch.q_lora_qr, &mut extra.q_lora_qr, XId::QLoraQr)?;
    copy_dense(
        &batch.attn_out_grouped,
        &mut extra.attn_out_grouped,
        XId::AttnOutGrouped,
    )?;
    copy_dense(&batch.o_lora, &mut extra.o_lora, XId::OLora)?;
    copy_dense(
        &batch.swiglu_hidden_shared,
        &mut extra.swiglu_hidden_shared,
        XId::SwigluHiddenShared,
    )?;
    copy_dense(
        &batch.hc_flat_pre_attn,
        &mut extra.hc_flat_pre_attn,
        XId::HcFlatPreAttn,
    )?;
    copy_dense(
        &batch.hc_flat_pre_ffn,
        &mut extra.hc_flat_pre_ffn,
        XId::HcFlatPreFfn,
    )?;
    copy_dense(&batch.h_final, &mut extra.h_final, XId::HFinal)?;
    if set.includes(XId::SwigluHiddenRouted) {
        if let Some(per_token) = batch.swiglu_hidden_routed.as_ref() {
            let width = geometry.dim_of(XId::SwigluHiddenRouted);
            let mut kept = Vec::new();
            for &(eid, ref row) in &per_token[t] {
                if row.len() != width {
                    return Err(model_err(format!(
                        "swiglu_hidden_routed E{eid} token {t}: {} != {width}",
                        row.len()
                    )));
                }
                if selected.contains(&eid) {
                    kept.push((eid, row.clone()));
                }
            }
            extra.swiglu_hidden_routed = kept;
        }
    }
    Ok(())
}

/// Worst-case unique retained rows at one layer under first-N (no multi-route credit).
#[inline]
pub fn worst_case_unique_rows_per_layer(
    geometry: &CaptureGeometry,
    max_hidden_tokens_per_expert: usize,
) -> usize {
    max_hidden_tokens_per_expert.saturating_mul(geometry.routed_experts)
}

/// Worst-case resident retained-hidden bytes at one layer for the doctor6 row.
#[inline]
pub fn worst_case_retained_hidden_bytes_per_layer(
    geometry: &CaptureGeometry,
    max_hidden_tokens_per_expert: usize,
) -> usize {
    worst_case_unique_rows_per_layer(geometry, max_hidden_tokens_per_expert)
        .saturating_mul(geometry.hidden)
        .saturating_mul(4)
}

pub fn format_capture_progress(
    probe_count: usize,
    total_tokens: usize,
    geometry: &CaptureGeometry,
    max_hidden_tokens_per_expert: usize,
) -> String {
    let worst =
        worst_case_unique_rows_per_layer(geometry, max_hidden_tokens_per_expert).min(total_tokens);
    let mib = (worst.saturating_mul(geometry.hidden).saturating_mul(4) as f64) / (1024.0 * 1024.0);
    format!(
        "dsv4f capture: {probe_count} probes, {total_tokens} tokens, {} layers, {} experts, \
         per-expert first-N={max_hidden_tokens_per_expert} (worst-case {worst} rows/layer \
         ≈{mib:.1} MiB f32@{})",
        geometry.layers, geometry.routed_experts, geometry.hidden
    )
}

/// On-disk relative path for the doctor6 router-input row.
///
/// Matches Q80: `hidden/L{{layer}}/{{probe}}/{{pos}}.f32le`.
pub fn retained_hidden_relative_path(layer: usize, probe_id: &str, position: usize) -> String {
    format!("hidden/L{layer:02}/{probe_id}/{position:06}.f32le")
}

pub fn retained_x_relative_path(
    x_id: &str,
    layer: usize,
    probe_id: &str,
    position: usize,
) -> String {
    format!("x/{x_id}/L{layer:02}/{probe_id}/{position:06}.f32le")
}

pub fn retained_routed_swiglu_relative_path(
    layer: usize,
    expert: u32,
    probe_id: &str,
    position: usize,
) -> String {
    format!("x/swiglu_hidden_routed/L{layer:02}/E{expert:03}/{probe_id}/{position:06}.f32le")
}

fn refuse_bad_probe_id(probe_id: &str) -> Result<()> {
    if probe_id.is_empty()
        || probe_id.contains('/')
        || probe_id.contains('\\')
        || probe_id.contains("..")
    {
        return Err(model_err(format!(
            "refusing probe_id {probe_id:?}: must be a single path component"
        )));
    }
    Ok(())
}

/// Write one retained row as little-endian f32. Refuses to overwrite.
pub fn write_retained_hidden_f32le(path: &Path, values: &[f32]) -> Result<(String, usize)> {
    if values.is_empty() {
        return Err(model_err(format!(
            "refusing to write empty hidden row at {}",
            path.display()
        )));
    }
    let parent = path.parent().ok_or_else(|| {
        model_err(format!(
            "hidden capture path has no parent: {}",
            path.display()
        ))
    })?;
    fs::create_dir_all(parent).map_err(|e| {
        model_err(format!(
            "cannot create hidden capture directory {}: {e}",
            parent.display()
        ))
    })?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| model_err(format!("cannot create hidden {}: {e}", path.display())))?;
    let mut digest = Sha256::new();
    for value in values {
        let bytes = value.to_le_bytes();
        file.write_all(&bytes)
            .map_err(|e| model_err(format!("cannot write hidden {}: {e}", path.display())))?;
        digest.update(bytes);
    }
    file.flush()
        .map_err(|e| model_err(format!("cannot flush hidden {}: {e}", path.display())))?;
    Ok((format!("{:x}", digest.finalize()), values.len() * 4))
}

/// Read a float32-LE row and refuse a short or corrupt file.
///
/// A file whose byte length is not `expected_elements * 4` is an error. The
/// reader never silently truncates or pads.
pub fn read_retained_hidden_f32le(path: &Path, expected_elements: usize) -> Result<Vec<f32>> {
    let mut file = fs::File::open(path)
        .map_err(|e| model_err(format!("cannot open hidden {}: {e}", path.display())))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| model_err(format!("cannot read hidden {}: {e}", path.display())))?;
    if bytes.len() % 4 != 0 {
        return Err(model_err(format!(
            "corrupt hidden {}: {} bytes is not a multiple of 4",
            path.display(),
            bytes.len()
        )));
    }
    let n = bytes.len() / 4;
    if n != expected_elements {
        return Err(model_err(format!(
            "hidden size mismatch at {}: got {n} f32 elements, expected {expected_elements}",
            path.display()
        )));
    }
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// One on-disk row recorded during flush.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HiddenWrite {
    pub relative_path: String,
    pub sha256: String,
    pub bytes: usize,
    pub elements: usize,
}

impl HiddenWrite {
    pub fn to_json(&self) -> Value {
        json!({
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "elements": self.elements,
        })
    }
}

/// Flush metadata for one retained token at one layer.
#[derive(Clone, Debug, Default)]
pub struct TokenFlushRecord {
    pub router_input: Option<HiddenWrite>,
    pub extra: BTreeMap<String, HiddenWrite>,
    pub routed_swiglu: Vec<(u32, HiddenWrite)>,
}

#[derive(Clone, Debug, Default)]
pub struct FlushBook {
    /// `(probe, position, layer) -> written rows`.
    pub rows: BTreeMap<(usize, usize, usize), TokenFlushRecord>,
}

#[derive(Clone, Debug, Default)]
pub struct LayerWriteStats {
    pub hidden_rows: usize,
    pub hidden_bytes: usize,
}

fn write_one(output_dir: &Path, rel: String, values: &[f32]) -> Result<HiddenWrite> {
    let (sha256, bytes) = write_retained_hidden_f32le(&output_dir.join(&rel), values)?;
    Ok(HiddenWrite {
        relative_path: rel,
        sha256,
        bytes,
        elements: values.len(),
    })
}

/// Write this layer's retained rows and record paths/hashes in `book`.
pub fn write_layer_retained_rows(
    output_dir: &Path,
    probes: &[(String, Vec<u32>)],
    layer_idx: usize,
    captures: &[Vec<Vec<LayerTokenCapture>>],
    geometry: &CaptureGeometry,
    book: &mut FlushBook,
) -> Result<LayerWriteStats> {
    let mut stats = LayerWriteStats::default();
    for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
        refuse_bad_probe_id(probe_id)?;
        for pos in 0..token_ids.len() {
            let cap = captures
                .get(pi)
                .and_then(|p| p.get(pos))
                .and_then(|t| t.iter().find(|c| c.layer == layer_idx))
                .ok_or_else(|| {
                    model_err(format!(
                        "flush missing capture {probe_id}@{pos} L{layer_idx}"
                    ))
                })?;
            if !cap.hidden_retained {
                continue;
            }
            if cap.router_input_hidden.len() != geometry.hidden {
                return Err(model_err(format!(
                    "{probe_id}@{pos} L{layer_idx}: retained hidden width {} != {}",
                    cap.router_input_hidden.len(),
                    geometry.hidden
                )));
            }
            let mut rec = TokenFlushRecord::default();
            rec.router_input = Some(write_one(
                output_dir,
                retained_hidden_relative_path(layer_idx, probe_id, pos),
                &cap.router_input_hidden,
            )?);
            stats.hidden_rows = stats.hidden_rows.saturating_add(1);
            stats.hidden_bytes = stats
                .hidden_bytes
                .saturating_add(rec.router_input.as_ref().map(|w| w.bytes).unwrap_or(0));
            let mut put_extra = |name: &str, values: &[f32]| -> Result<()> {
                if values.is_empty() {
                    return Ok(());
                }
                rec.extra.insert(
                    name.to_string(),
                    write_one(
                        output_dir,
                        retained_x_relative_path(name, layer_idx, probe_id, pos),
                        values,
                    )?,
                );
                Ok(())
            };
            put_extra("h_post_attn_norm", &cap.extra_x.h_post_attn_norm)?;
            put_extra("q_lora_qr", &cap.extra_x.q_lora_qr)?;
            put_extra("attn_out_grouped", &cap.extra_x.attn_out_grouped)?;
            put_extra("o_lora", &cap.extra_x.o_lora)?;
            put_extra("swiglu_hidden_shared", &cap.extra_x.swiglu_hidden_shared)?;
            put_extra("hc_flat_pre_attn", &cap.extra_x.hc_flat_pre_attn)?;
            put_extra("hc_flat_pre_ffn", &cap.extra_x.hc_flat_pre_ffn)?;
            put_extra("h_final", &cap.extra_x.h_final)?;
            for &(eid, ref row) in &cap.extra_x.swiglu_hidden_routed {
                rec.routed_swiglu.push((
                    eid,
                    write_one(
                        output_dir,
                        retained_routed_swiglu_relative_path(layer_idx, eid, probe_id, pos),
                        row,
                    )?,
                ));
            }
            book.rows.insert((pi, pos, layer_idx), rec);
        }
    }
    Ok(stats)
}

fn percentile_sorted(sorted: &[usize], p: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn dist_block(counts: &[usize], row_threshold: usize) -> Value {
    let mut sorted = counts.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let below_8 = sorted.iter().filter(|&&c| c < 8).count();
    let below_16 = sorted.iter().filter(|&&c| c < 16).count();
    let below_32 = sorted.iter().filter(|&&c| c < 32).count();
    let at_or_above_64 = sorted.iter().filter(|&&c| c >= 64).count();
    let at_or_above_thr = sorted.iter().filter(|&&c| c >= row_threshold).count();
    let zero = sorted.iter().filter(|&&c| c == 0).count();
    let mean = if n == 0 {
        0.0
    } else {
        counts.iter().sum::<usize>() as f64 / n as f64
    };
    let frac = |c: usize| if n == 0 { 0.0 } else { c as f64 / n as f64 };
    json!({
        "n_pairs": n,
        "p10": percentile_sorted(&sorted, 10.0),
        "p50": percentile_sorted(&sorted, 50.0),
        "p90": percentile_sorted(&sorted, 90.0),
        "min": sorted.first().copied().unwrap_or(0),
        "max": sorted.last().copied().unwrap_or(0),
        "mean": mean,
        "frac_below_8": frac(below_8),
        "frac_below_16": frac(below_16),
        "frac_below_32": frac(below_32),
        "frac_at_or_above_64": frac(at_or_above_64),
        "frac_at_or_above_row_threshold": frac(at_or_above_thr),
        "pct_zero": if n == 0 { 0.0 } else { 100.0 * zero as f64 / n as f64 },
        "count_below_8": below_8,
        "count_below_16": below_16,
        "count_below_32": below_32,
        "count_at_or_above_64": at_or_above_64,
        "count_at_or_above_row_threshold": at_or_above_thr,
        "count_zero": zero,
        "row_threshold": row_threshold,
    })
}

/// Per-(layer, expert) retained-hidden hit counts — the quantity organs fit on.
///
/// Mirrors Q80's `n_fit_distribution`: a retained token contributes one row to
/// every expert in its top-k for that layer. After flush, `hidden_retained` is
/// the authority (payloads are gone).
pub fn n_fit_distribution(
    captures: &[Vec<Vec<LayerTokenCapture>>],
    geometry: &CaptureGeometry,
    row_threshold: usize,
) -> Value {
    let mut by_layer_expert: Vec<usize> =
        Vec::with_capacity(geometry.layers.saturating_mul(geometry.routed_experts));
    let mut by_layer: Vec<usize> = Vec::with_capacity(geometry.layers);
    let mut by_expert: Vec<usize> = vec![0usize; geometry.routed_experts];
    let mut shared_per_layer: Vec<usize> = Vec::with_capacity(geometry.layers);
    for layer in 0..geometry.layers {
        let mut per_expert = vec![0usize; geometry.routed_experts];
        let mut shared = 0usize;
        for probe_caps in captures {
            for token_caps in probe_caps {
                let Some(layer_cap) = token_caps.iter().find(|c| c.layer == layer) else {
                    continue;
                };
                if !layer_cap.hidden_retained {
                    continue;
                }
                shared += 1;
                for &expert in &layer_cap.selected_expert_ids {
                    let e = expert as usize;
                    if e < geometry.routed_experts {
                        per_expert[e] += 1;
                    }
                }
            }
        }
        let layer_total: usize = per_expert.iter().sum();
        by_layer.push(layer_total);
        shared_per_layer.push(shared);
        for e in 0..geometry.routed_experts {
            by_expert[e] += per_expert[e];
        }
        by_layer_expert.extend(per_expert);
    }
    let mut overall = dist_block(&by_layer_expert, row_threshold);
    overall["unit"] = json!("retained_hidden_rows_per_layer_expert");
    overall["n_layer_expert_pairs"] = json!(by_layer_expert.len());
    overall["experts"] = json!(geometry.routed_experts);
    overall["layers"] = json!(geometry.layers);
    json!({
        "unit": "retained_hidden_rows_per_layer_expert",
        "n_layer_expert_pairs": by_layer_expert.len(),
        "experts": geometry.routed_experts,
        "layers": geometry.layers,
        "p10": overall["p10"],
        "p50": overall["p50"],
        "p90": overall["p90"],
        "min": overall["min"],
        "max": overall["max"],
        "mean": overall["mean"],
        "frac_below_8": overall["frac_below_8"],
        "frac_below_16": overall["frac_below_16"],
        "frac_below_32": overall["frac_below_32"],
        "frac_at_or_above_64": overall["frac_at_or_above_64"],
        "frac_at_or_above_row_threshold": overall["frac_at_or_above_row_threshold"],
        "pct_zero": overall["pct_zero"],
        "count_below_8": overall["count_below_8"],
        "count_below_16": overall["count_below_16"],
        "count_below_32": overall["count_below_32"],
        "count_at_or_above_64": overall["count_at_or_above_64"],
        "count_at_or_above_row_threshold": overall["count_at_or_above_row_threshold"],
        "count_zero": overall["count_zero"],
        "row_threshold": row_threshold,
        "by_layer_totals": dist_block(&by_layer, row_threshold),
        "by_expert_totals_across_layers": dist_block(&by_expert, row_threshold),
        "shared_expert": {
            "unit": "retained_hidden_rows_per_layer",
            "note": "Shared expert sees every token; n_fit equals retained tokens at the layer. Same X as routed w1 (h_post_ffn_norm).",
            "per_layer": shared_per_layer,
            "distribution": dist_block(&shared_per_layer, row_threshold),
        },
        "by_component": {
            "note": "routed w1/w3 and router_gate share h_post_ffn_norm; n_fit is identical across those organs. routed w2 uses swiglu_hidden_routed.",
            "routed_expert.w1": "same_as_overall",
            "routed_expert.w3": "same_as_overall",
            "router_gate.weight": "same_as_overall",
            "shared_expert.w1": "same_as_shared_expert",
            "shared_expert.w3": "same_as_shared_expert",
        },
    })
}

pub fn empty_captures(probes: &[(String, Vec<u32>)]) -> Vec<Vec<Vec<LayerTokenCapture>>> {
    probes
        .iter()
        .map(|(_, toks)| (0..toks.len()).map(|_| Vec::new()).collect())
        .collect()
}

pub fn build_token_index(probes: &[(String, Vec<u32>)]) -> Vec<(usize, usize)> {
    let mut idx = Vec::new();
    for (pi, (_, toks)) in probes.iter().enumerate() {
        for pos in 0..toks.len() {
            idx.push((pi, pos));
        }
    }
    idx
}

/// How synthetic (and fairness) routes are generated.
#[derive(Clone, Debug)]
pub enum RoutePlan {
    /// Token `t` cycles through disjoint top-k windows modulo `routed_experts`.
    Cycling,
    /// Most tokens hit `common`; every `rare_period`-th token also includes `rare`.
    Skewed {
        common: Vec<u32>,
        rare: u32,
        rare_period: usize,
    },
}

fn unique_route(preferred: &[u32], top_k: usize, n_experts: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(top_k);
    let mut seen = vec![false; n_experts];
    for &e in preferred {
        let eu = e as usize;
        if eu < n_experts && !seen[eu] && out.len() < top_k {
            seen[eu] = true;
            out.push(e);
        }
    }
    for e in 0..n_experts {
        if out.len() >= top_k {
            break;
        }
        if !seen[e] {
            seen[e] = true;
            out.push(e as u32);
        }
    }
    out
}

pub fn route_for_token(
    plan: &RoutePlan,
    token: usize,
    geometry: &CaptureGeometry,
) -> (Vec<u32>, Vec<f32>) {
    let ids = match plan {
        RoutePlan::Cycling => {
            let start = token.saturating_mul(geometry.top_k);
            (0..geometry.top_k)
                .map(|k| ((start + k) % geometry.routed_experts.max(1)) as u32)
                .collect()
        }
        RoutePlan::Skewed {
            common,
            rare,
            rare_period,
        } => {
            let mut preferred = Vec::new();
            if *rare_period > 0 && token % *rare_period == 0 {
                preferred.push(*rare);
            }
            preferred.extend(common.iter().copied());
            unique_route(&preferred, geometry.top_k, geometry.routed_experts)
        }
    };
    let w = if geometry.top_k == 0 {
        0.0
    } else {
        1.0 / geometry.top_k as f32
    };
    let weights = vec![w; ids.len()];
    (ids, weights)
}

fn fill_row(out: &mut [f32], layer: usize, token: usize, tag: u32) {
    for (i, x) in out.iter_mut().enumerate() {
        *x = (layer as f32) + (token as f32) * 1.0e-3 + (i as f32) * 1.0e-6 + (tag as f32) * 1.0e-2;
    }
}

fn packed_dense(n: usize, width: usize, layer: usize, tag: u32) -> Vec<f32> {
    let mut buf = vec![0.0f32; n.saturating_mul(width)];
    for t in 0..n {
        fill_row(&mut buf[t * width..(t + 1) * width], layer, t, tag);
    }
    buf
}

/// Deterministic synthetic routes + X for one layer. Same inputs ⇒ same bytes.
pub fn synthetic_batch_for_layer(
    token_count: usize,
    layer_idx: usize,
    geometry: &CaptureGeometry,
    set: &CaptureSet,
    plan: &RoutePlan,
) -> (Vec<(Vec<u32>, Vec<f32>)>, LayerActivationBatch) {
    let routes: Vec<(Vec<u32>, Vec<f32>)> = (0..token_count)
        .map(|t| route_for_token(plan, t, geometry))
        .collect();
    let mut batch = LayerActivationBatch {
        h_post_ffn_norm: packed_dense(token_count, geometry.hidden, layer_idx, 1),
        ..LayerActivationBatch::default()
    };
    if set.includes(XId::HPostAttnNorm) {
        batch.h_post_attn_norm = Some(packed_dense(token_count, geometry.hidden, layer_idx, 2));
    }
    if set.includes(XId::QLoraQr) {
        batch.q_lora_qr = Some(packed_dense(token_count, geometry.q_lora, layer_idx, 3));
    }
    if set.includes(XId::AttnOutGrouped) {
        batch.attn_out_grouped = Some(packed_dense(token_count, geometry.hidden, layer_idx, 4));
    }
    if set.includes(XId::OLora) {
        batch.o_lora = Some(packed_dense(token_count, geometry.o_lora, layer_idx, 5));
    }
    if set.includes(XId::SwigluHiddenShared) {
        batch.swiglu_hidden_shared = Some(packed_dense(
            token_count,
            geometry.moe_intermediate,
            layer_idx,
            6,
        ));
    }
    if set.includes(XId::SwigluHiddenRouted) {
        let width = geometry.moe_intermediate;
        let mut per_token = Vec::with_capacity(token_count);
        for (t, (ids, _)) in routes.iter().enumerate() {
            let mut rows = Vec::with_capacity(ids.len());
            for &eid in ids {
                let mut row = vec![0.0f32; width];
                fill_row(&mut row, layer_idx, t, 100 + eid);
                rows.push((eid, row));
            }
            per_token.push(rows);
        }
        batch.swiglu_hidden_routed = Some(per_token);
    }
    if set.includes(XId::HcFlatPreAttn) {
        batch.hc_flat_pre_attn = Some(packed_dense(token_count, geometry.hc_flat, layer_idx, 7));
    }
    if set.includes(XId::HcFlatPreFfn) {
        batch.hc_flat_pre_ffn = Some(packed_dense(token_count, geometry.hc_flat, layer_idx, 8));
    }
    if set.includes(XId::HFinal) && layer_idx + 1 == DSV4F_CAPTURE_LAYERS {
        batch.h_final = Some(packed_dense(token_count, geometry.hidden, layer_idx, 9));
    }
    (routes, batch)
}

pub fn write_json_new(path: &Path, value: &Value) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| model_err(format!("json path has no parent: {}", path.display())))?;
    fs::create_dir_all(parent)
        .map_err(|e| model_err(format!("cannot create {}: {e}", parent.display())))?;
    let text = serde_json::to_string_pretty(value)
        .map_err(|e| model_err(format!("cannot serialize {}: {e}", path.display())))?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| model_err(format!("cannot create {}: {e}", path.display())))?;
    file.write_all(text.as_bytes())
        .map_err(|e| model_err(format!("cannot write {}: {e}", path.display())))?;
    file.write_all(b"\n")
        .map_err(|e| model_err(format!("cannot finish {}: {e}", path.display())))?;
    file.flush()
        .map_err(|e| model_err(format!("cannot flush {}: {e}", path.display())))?;
    Ok(())
}

/// Emit the Q80/doctor6 `capture-result.json` plus DSV4F organ extras.
pub fn emit_capture_result(
    output_dir: &Path,
    probes: &[(String, Vec<u32>)],
    captures: &[Vec<Vec<LayerTokenCapture>>],
    book: &FlushBook,
    geometry: &CaptureGeometry,
    set: &CaptureSet,
    max_per_expert: usize,
    hidden_rows_retained_total: usize,
    hidden_rows_per_layer: &[usize],
    hidden_bytes_written: usize,
    n_fit: &Value,
) -> Result<PathBuf> {
    let mut probe_rows = Vec::with_capacity(probes.len());
    let mut tokens_executed = 0usize;
    let mut route_membership_total = 0usize;
    for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
        let mut steps = Vec::with_capacity(token_ids.len());
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let layer_caps = &captures[pi][pos];
            if layer_caps.len() != geometry.layers {
                return Err(model_err(format!(
                    "{probe_id}@{pos}: captured {} layers, expected {}",
                    layer_caps.len(),
                    geometry.layers
                )));
            }
            let mut layer_rows = Vec::with_capacity(geometry.layers);
            let mut any_retained = false;
            for layer_cap in layer_caps {
                if layer_cap.selected_expert_ids.len() != geometry.top_k
                    || layer_cap.normalized_route_weights.len() != geometry.top_k
                {
                    return Err(model_err(format!(
                        "{probe_id}@{pos} L{}: route membership is not top-{}",
                        layer_cap.layer, geometry.top_k
                    )));
                }
                route_membership_total =
                    route_membership_total.saturating_add(layer_cap.selected_expert_ids.len());
                let store_hidden = layer_cap.hidden_retained;
                if store_hidden {
                    any_retained = true;
                }
                let rec = book.rows.get(&(pi, pos, layer_cap.layer));
                let hidden_meta = if store_hidden {
                    let written = rec.and_then(|r| r.router_input.as_ref()).ok_or_else(|| {
                        model_err(format!(
                            "{probe_id}@{pos} L{}: retained but not written during flush",
                            layer_cap.layer
                        ))
                    })?;
                    Some(json!({
                        "relative_path": written.relative_path,
                        "sha256": written.sha256,
                        "bytes": written.bytes,
                        "elements": written.elements,
                        "source": "DSV4F h_post_ffn_norm (router-input / routed+shared w1+w3 X)",
                    }))
                } else {
                    None
                };
                let mut x_matrices = json!({});
                if let Some(rec) = rec {
                    if let Some(obj) = x_matrices.as_object_mut() {
                        for (name, w) in &rec.extra {
                            obj.insert(name.clone(), w.to_json());
                        }
                        if !rec.routed_swiglu.is_empty() {
                            obj.insert(
                                "swiglu_hidden_routed".into(),
                                json!(rec
                                    .routed_swiglu
                                    .iter()
                                    .map(|(eid, w)| {
                                        json!({
                                            "expert_id": eid,
                                            "relative_path": w.relative_path,
                                            "sha256": w.sha256,
                                            "bytes": w.bytes,
                                            "elements": w.elements,
                                        })
                                    })
                                    .collect::<Vec<_>>()),
                            );
                        }
                    }
                }
                layer_rows.push(json!({
                    "layer": layer_cap.layer,
                    "selected_expert_ids": layer_cap.selected_expert_ids,
                    "normalized_route_weights": layer_cap.normalized_route_weights,
                    "router_input_hidden_f32le": hidden_meta,
                    "hidden_retained": store_hidden,
                    "x_matrices": x_matrices,
                }));
            }
            steps.push(json!({
                "position": pos,
                "input_token_id": token_id,
                "layers": layer_rows,
                "hidden_retained_for_this_token": any_retained,
            }));
            tokens_executed += 1;
        }
        probe_rows.push(json!({
            "probe_id": probe_id,
            "source_one_user_native_prompt_token_count": steps.len(),
            "steps": steps,
        }));
    }
    let expected_route_slots = tokens_executed
        .saturating_mul(geometry.layers)
        .saturating_mul(geometry.top_k);
    if route_membership_total != expected_route_slots {
        return Err(model_err(format!(
            "route membership total {route_membership_total} != expected {expected_route_slots}"
        )));
    }
    let organs_json: Vec<Value> = CAPTURE_ORGANS
        .iter()
        .map(|o| {
            json!({
                "organ": o.name,
                "input_dim": o.input_dim,
                "x_id": o.x_id.as_str(),
                "required_for_1_5_complete": o.required_for_1_5_complete,
            })
        })
        .collect();
    let result = json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_DSV4F_ACTIVATION_X_CAPTURE",
        "capture_protocol_revision": CAPTURE_PROTOCOL_REVISION,
        "runtime_binding": {
            "architecture": "deepseek_v4_flash",
            "weight_backend": "writer_only_activations_supplied_by_caller",
            "metal_not_used": true,
            "layers": geometry.layers,
            "hidden": geometry.hidden,
            "experts": geometry.routed_experts,
            "shared_experts": DSV4F_CAPTURE_SHARED_EXPERTS,
            "top_k": geometry.top_k,
            "moe_intermediate": geometry.moe_intermediate,
            "q_lora": geometry.q_lora,
            "o_lora": geometry.o_lora,
        },
        "activation_x_capture": {
            "required_count": DSV4F_CAPTURE_REQUIRED_ORGANS,
            "organs": organs_json,
            "capture_set": {
                "h_post_ffn_norm": true,
                "h_post_attn_norm": set.h_post_attn_norm,
                "q_lora_qr": set.q_lora_qr,
                "attn_out_grouped": set.attn_out_grouped,
                "o_lora": set.o_lora,
                "swiglu_hidden_routed": set.swiglu_hidden_routed,
                "swiglu_hidden_shared": set.swiglu_hidden_shared,
                "optional_mhc": set.optional_mhc,
                "optional_lm_head": set.optional_lm_head,
            },
        },
        "bounded_storage": {
            "strategy": "per_expert_first_n_router_input_hiddens_plus_full_route_membership",
            "why": "A shared per-layer budget starves 256 routed experts. First-N per expert after routing is known guarantees up to N retained rows per (layer, expert). Rows are flushed per layer and freed before the next layer loads.",
            "max_hidden_tokens_per_expert": max_per_expert,
            "retention_policy": "first_N_tokens_that_route_to_expert_in_global_token_order",
            "deterministic": true,
            "experts": geometry.routed_experts,
            "worst_case_unique_rows_per_layer": worst_case_unique_rows_per_layer(geometry, max_per_expert),
            "hidden_rows_retained_total": hidden_rows_retained_total,
            "hidden_rows_retained_per_layer": hidden_rows_per_layer,
            "layers": geometry.layers,
            "hidden_width": geometry.hidden,
            "total_tokens_executed": tokens_executed,
            "retained_hidden_bytes_written": hidden_bytes_written,
            "full_route_membership_for_every_token_every_layer": true,
            "n_fit_distribution": n_fit,
            "rejected_alternatives": {
                "full_raw_all_tokens": "unbounded; not acceptable",
                "per_layer_stratified_subsample": "shared budget across 256 experts; larger corpus spreads routing and starves each expert",
                "random_reservoir": "not deterministic unless seeded and documented; first-N is byte-identical across runs",
            },
        },
        "capture_summary": {
            "probe_count": probe_rows.len(),
            "total_tokens": tokens_executed,
            "layers_executed": geometry.layers,
            "all_layer_activation_capture": true,
            "hidden_rows_retained_total": hidden_rows_retained_total,
            "max_hidden_tokens_per_expert": max_per_expert,
            "n_fit_distribution": n_fit,
        },
        "probes": probe_rows,
        "claim_boundary": {
            "new_diagnostic_not_historical": true,
            "diagnostic_activation_pricing_only": true,
            "bounded_hidden_storage_not_unbounded_raw_dump": true,
            "per_expert_first_n_retention": true,
            "doctor6_collect_expert_activations_shape": true,
            "full_43_layer_source_forward_not_executed_by_this_writer": true,
            "metal_sparse_attention_indexer_graph_not_implemented": true,
            "layers_ge_2_of_a_real_source_need_metal_sparse_attention_or_a_cpu_streamed_substitute": true,
            "does_not_claim_COMPLETE_PHYSICAL_BPW_coherence_hcli_or_tps": true,
        },
    });
    let path = output_dir.join("capture-result.json");
    write_json_new(&path, &result)?;
    Ok(path)
}

/// Request a reduced (or fixture) capture. Never opens the sealed artifact.
#[derive(Clone, Debug)]
pub struct ReducedCaptureRequest {
    pub output_dir: PathBuf,
    pub geometry: CaptureGeometry,
    pub set: CaptureSet,
    pub probes: Vec<(String, Vec<u32>)>,
    pub max_per_expert: usize,
    pub row_threshold: usize,
    pub route_plan: RoutePlan,
}

#[derive(Clone, Debug)]
pub struct ReducedCaptureReport {
    pub result_path: PathBuf,
    pub hidden_rows_retained_total: usize,
    pub hidden_bytes_written: usize,
    pub peak_resident_bytes: usize,
    pub resident_after_append: Vec<usize>,
    pub n_fit: Value,
}

/// Layer-major retain → flush → free over a synthetic (or caller-built) corpus.
///
/// Peak retained-row RAM is one layer. After each layer's flush the payloads
/// are dropped before the next layer is materialized.
pub fn run_reduced_capture(req: &ReducedCaptureRequest) -> Result<ReducedCaptureReport> {
    if req.output_dir.exists() {
        return Err(model_err(format!(
            "refusing to reuse or overwrite capture output directory {}",
            req.output_dir.display()
        )));
    }
    if req.max_per_expert == 0 {
        return Err(model_err(
            "max_hidden_tokens_per_expert must be > 0; a zero quota retains nothing",
        ));
    }
    if req.geometry.top_k == 0 || req.geometry.routed_experts == 0 || req.geometry.layers == 0 {
        return Err(model_err("geometry layers/experts/top_k must be > 0"));
    }
    fs::create_dir_all(&req.output_dir)
        .map_err(|e| model_err(format!("cannot create {}: {e}", req.output_dir.display())))?;
    let token_index = build_token_index(&req.probes);
    let total_tokens = token_index.len();
    let mut captures = empty_captures(&req.probes);
    let mut book = FlushBook::default();
    let mut hidden_rows_retained_total = 0usize;
    let mut hidden_bytes_written = 0usize;
    let mut hidden_rows_per_layer = vec![0usize; req.geometry.layers];
    let mut resident_after_append = Vec::with_capacity(req.geometry.layers);
    let mut peak_resident_bytes = 0usize;

    for layer_idx in 0..req.geometry.layers {
        let (mut routes, batch) = synthetic_batch_for_layer(
            total_tokens,
            layer_idx,
            &req.geometry,
            &req.set,
            &req.route_plan,
        );
        append_retained_layer_captures(
            &mut captures,
            &token_index,
            &mut routes,
            &batch,
            layer_idx,
            &req.geometry,
            &req.set,
            req.max_per_expert,
        )?;
        let resident = resident_retained_hidden_bytes(&captures);
        resident_after_append.push(resident);
        peak_resident_bytes = peak_resident_bytes.max(resident);
        // Only this layer may hold hidden payloads.
        for probe in &captures {
            for token in probe {
                for cap in token {
                    if cap.layer != layer_idx
                        && (!cap.router_input_hidden.is_empty() || cap.extra_x.f32_elements() > 0)
                    {
                        return Err(model_err(format!(
                            "layer {} still resident while appending layer {layer_idx}",
                            cap.layer
                        )));
                    }
                }
            }
        }
        let stats = write_layer_retained_rows(
            &req.output_dir,
            &req.probes,
            layer_idx,
            &captures,
            &req.geometry,
            &mut book,
        )?;
        hidden_rows_retained_total = hidden_rows_retained_total.saturating_add(stats.hidden_rows);
        hidden_bytes_written = hidden_bytes_written.saturating_add(stats.hidden_bytes);
        if layer_idx < hidden_rows_per_layer.len() {
            hidden_rows_per_layer[layer_idx] = stats.hidden_rows;
        }
        release_layer_retained_hiddens(&mut captures, layer_idx);
        if resident_retained_hidden_bytes(&captures) != 0 {
            return Err(model_err(format!(
                "layer {layer_idx} hiddens must be freed before the next layer loads"
            )));
        }
    }

    let n_fit = n_fit_distribution(&captures, &req.geometry, req.row_threshold);
    let result_path = emit_capture_result(
        &req.output_dir,
        &req.probes,
        &captures,
        &book,
        &req.geometry,
        &req.set,
        req.max_per_expert,
        hidden_rows_retained_total,
        &hidden_rows_per_layer,
        hidden_bytes_written,
        &n_fit,
    )?;
    Ok(ReducedCaptureReport {
        result_path,
        hidden_rows_retained_total,
        hidden_bytes_written,
        peak_resident_bytes,
        resident_after_append,
        n_fit,
    })
}

pub fn default_synthetic_probes(
    n_probes: usize,
    tokens_per_probe: usize,
) -> Vec<(String, Vec<u32>)> {
    (0..n_probes)
        .map(|i| {
            let id = format!("probe{i}");
            let toks = (0..tokens_per_probe)
                .map(|t| 1000 + (i * 10_000 + t) as u32)
                .collect();
            (id, toks)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn sealed_organ_catalog_matches_schedule_receipt() {
        let required: Vec<_> = required_organs().collect();
        assert_eq!(required.len(), DSV4F_CAPTURE_REQUIRED_ORGANS);
        assert_eq!(COMPRESS_RATIOS.len(), 44);
        assert_eq!(compress_ratio(0), Some(0));
        assert_eq!(compress_ratio(1), Some(0));
        assert_eq!(compress_ratio(2), Some(4));
        assert_eq!(compress_ratio(3), Some(128));
        assert_eq!(compress_ratio(42), Some(4));

        let receipt = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../receipts/DSV4F_TENSOR_SCHEDULE.json");
        let text = std::fs::read_to_string(&receipt).expect("sealed schedule receipt");
        let v: Value = serde_json::from_str(&text).expect("schedule json");
        let organs = v["activation_x_capture"]["organs"]
            .as_array()
            .expect("organs");
        let mut sealed_required = Vec::new();
        for row in organs {
            if row["required_for_1_5_complete"].as_bool() == Some(true) {
                sealed_required.push((
                    row["organ"].as_str().unwrap().to_string(),
                    row["input_dim"].as_u64().unwrap() as usize,
                ));
            }
        }
        let ours: Vec<(String, usize)> = required
            .iter()
            .map(|o| (o.name.to_string(), o.input_dim))
            .collect();
        assert_eq!(ours, sealed_required);
        assert!(organ_applies_to_layer(
            CAPTURE_ORGANS
                .iter()
                .find(|o| o.name == "indexer.wq_b")
                .unwrap(),
            2
        ));
        assert!(!organ_applies_to_layer(
            CAPTURE_ORGANS
                .iter()
                .find(|o| o.name == "indexer.wq_b")
                .unwrap(),
            0
        ));
        assert!(!organ_applies_to_layer(
            CAPTURE_ORGANS
                .iter()
                .find(|o| o.name == "compressor.wkv")
                .unwrap(),
            0
        ));
        assert!(organ_applies_to_layer(
            CAPTURE_ORGANS
                .iter()
                .find(|o| o.name == "compressor.wkv")
                .unwrap(),
            2
        ));
    }

    #[test]
    fn per_expert_first_n_retention_is_deterministic_and_bounded() {
        let max_n = 3usize;
        let mut counts = vec![0usize; DSV4F_CAPTURE_ROUTED_EXPERTS];
        let routes: Vec<Vec<u32>> = (0..20u32).map(|t| vec![t % 4, 4 + (t % 4)]).collect();
        let mut retained_mask = Vec::new();
        for ids in &routes {
            retained_mask.push(credit_expert_first_n_retention(&mut counts, ids, max_n));
        }
        for e in 0..8 {
            assert_eq!(counts[e], max_n, "expert {e} retained count");
        }
        for e in 8..DSV4F_CAPTURE_ROUTED_EXPERTS {
            assert_eq!(counts[e], 0, "untouched expert {e}");
        }
        assert!(retained_mask[0]);
        assert!(retained_mask[4]);
        assert!(retained_mask[8]);
        assert!(!retained_mask[12]);
        let mut counts2 = vec![0usize; DSV4F_CAPTURE_ROUTED_EXPERTS];
        let mask2: Vec<bool> = routes
            .iter()
            .map(|ids| credit_expert_first_n_retention(&mut counts2, ids, max_n))
            .collect();
        assert_eq!(retained_mask, mask2);
        assert_eq!(counts, counts2);
    }

    #[test]
    fn per_expert_first_n_zero_retains_nothing() {
        let mut counts = vec![0usize; DSV4F_CAPTURE_ROUTED_EXPERTS];
        let ids: Vec<u32> = (0..DSV4F_CAPTURE_TOP_K as u32).collect();
        assert!(!credit_expert_first_n_retention(&mut counts, &ids, 0));
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn rare_expert_is_not_starved_by_common_experts() {
        let g = CaptureGeometry {
            routed_experts: DSV4F_CAPTURE_ROUTED_EXPERTS,
            top_k: DSV4F_CAPTURE_TOP_K,
            ..CaptureGeometry::test_fixture()
        };
        let n = 5usize;
        let rare = 255u32;
        let plan = RoutePlan::Skewed {
            common: vec![0, 1, 2],
            rare,
            rare_period: 10,
        };
        let tokens = 200usize;
        let mut counts = vec![0usize; g.routed_experts];
        let mut rare_hits = 0usize;
        for t in 0..tokens {
            let (ids, _) = route_for_token(&plan, t, &g);
            assert_eq!(ids.len(), g.top_k);
            assert_eq!(ids.iter().copied().collect::<BTreeSet<_>>().len(), g.top_k);
            if ids.contains(&rare) {
                rare_hits += 1;
            }
            credit_expert_first_n_retention(&mut counts, &ids, n);
        }
        assert!(
            rare_hits >= n,
            "fixture must present the rare expert >= N times"
        );
        assert_eq!(
            counts[rare as usize], n,
            "rare expert {rare} must accumulate first-N rows, not be starved"
        );
        assert_eq!(counts[0], n);
        assert_eq!(counts[1], n);
        assert_eq!(counts[2], n);
        let global_budget_would_starve = tokens < n.saturating_mul(g.routed_experts);
        assert!(
            global_budget_would_starve,
            "this fixture is only meaningful if a global budget would be tight"
        );
    }

    #[test]
    fn retained_hiddens_do_not_accumulate_across_layers() {
        let g = CaptureGeometry::test_fixture();
        let set = CaptureSet::required();
        let token_count = 24usize;
        let n = 4usize;
        let probes = vec![("probe0".to_string(), vec![1u32; token_count])];
        let token_index = build_token_index(&probes);
        let mut captures = empty_captures(&probes);
        let mut after_append = Vec::with_capacity(g.layers);
        for layer_idx in 0..g.layers {
            let (mut routes, batch) =
                synthetic_batch_for_layer(token_count, layer_idx, &g, &set, &RoutePlan::Cycling);
            append_retained_layer_captures(
                &mut captures,
                &token_index,
                &mut routes,
                &batch,
                layer_idx,
                &g,
                &set,
                n,
            )
            .expect("append");
            let resident = resident_retained_hidden_bytes(&captures);
            after_append.push(resident);
            for probe in &captures {
                for token in probe {
                    for cap in token {
                        if cap.layer != layer_idx {
                            assert!(
                                cap.router_input_hidden.is_empty(),
                                "layer {} still resident while appending layer {layer_idx}",
                                cap.layer
                            );
                            assert_eq!(
                                cap.extra_x.f32_elements(),
                                0,
                                "layer {} extra X still resident while appending layer {layer_idx}",
                                cap.layer
                            );
                        }
                    }
                }
            }
            let freed = release_layer_retained_hiddens(&mut captures, layer_idx);
            assert!(freed > 0, "layer {layer_idx} should have retained rows");
            assert_eq!(
                resident_retained_hidden_bytes(&captures),
                0,
                "layer {layer_idx} hiddens must be freed before the next layer"
            );
            for token in &captures[0] {
                assert_eq!(token.len(), layer_idx + 1);
                let cap = &token[layer_idx];
                assert_eq!(cap.selected_expert_ids.len(), g.top_k);
                assert_eq!(cap.normalized_route_weights.len(), g.top_k);
                assert!(cap.router_input_hidden.is_empty());
                assert_eq!(cap.extra_x.f32_elements(), 0);
            }
        }
        let base = after_append[0];
        assert!(base > 0);
        for (layer_idx, &bytes) in after_append.iter().enumerate() {
            assert_eq!(
                bytes, base,
                "resident hidden bytes after append of layer {layer_idx} grew with L"
            );
        }
        assert!(
            base < g.layers.saturating_mul(base) / 2 + 1,
            "one-layer footprint must be far below the accumulated {}-layer total",
            g.layers
        );
    }

    #[test]
    fn same_probe_set_same_n_is_byte_identical_across_two_runs() {
        let g = CaptureGeometry::test_fixture();
        let set = CaptureSet::required();
        let probes = default_synthetic_probes(2, 12);
        let n = 3usize;
        let run = |dir: PathBuf| {
            run_reduced_capture(&ReducedCaptureRequest {
                output_dir: dir,
                geometry: g.clone(),
                set: set.clone(),
                probes: probes.clone(),
                max_per_expert: n,
                row_threshold: 2,
                route_plan: RoutePlan::Cycling,
            })
            .expect("capture")
        };
        let dir_a = tempfile::tempdir().expect("tempdir a");
        let dir_b = tempfile::tempdir().expect("tempdir b");
        let out_a = dir_a.path().join("run");
        let out_b = dir_b.path().join("run");
        let ra = run(out_a.clone());
        let rb = run(out_b.clone());
        assert_eq!(ra.hidden_rows_retained_total, rb.hidden_rows_retained_total);
        assert_eq!(ra.hidden_bytes_written, rb.hidden_bytes_written);
        assert_eq!(ra.resident_after_append, rb.resident_after_append);

        let collect = |root: &Path| -> Vec<(String, Vec<u8>)> {
            let mut files = Vec::new();
            fn walk(dir: &Path, root: &Path, files: &mut Vec<(String, Vec<u8>)>) {
                let Ok(rd) = std::fs::read_dir(dir) else {
                    return;
                };
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.is_dir() {
                        walk(&p, root, files);
                    } else if p.extension().and_then(|s| s.to_str()) == Some("f32le") {
                        let rel = p
                            .strip_prefix(root)
                            .expect("under root")
                            .to_string_lossy()
                            .into_owned();
                        files.push((rel, std::fs::read(p).expect("read")));
                    }
                }
            }
            walk(root, root, &mut files);
            files.sort_by(|a, b| a.0.cmp(&b.0));
            files
        };
        let files_a = collect(&out_a);
        let files_b = collect(&out_b);
        assert!(!files_a.is_empty(), "expected retained hidden files");
        assert_eq!(files_a, files_b);

        let ja = std::fs::read_to_string(out_a.join("capture-result.json")).unwrap();
        let jb = std::fs::read_to_string(out_b.join("capture-result.json")).unwrap();
        let va: Value = serde_json::from_str(&ja).unwrap();
        let vb: Value = serde_json::from_str(&jb).unwrap();
        assert_eq!(va["probes"], vb["probes"]);
        assert_eq!(
            va["bounded_storage"]["n_fit_distribution"],
            vb["bounded_storage"]["n_fit_distribution"]
        );
    }

    #[test]
    fn short_or_corrupt_row_raises_rather_than_truncating() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("row.f32le");
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        write_retained_hidden_f32le(&path, &values).expect("write");
        let got = read_retained_hidden_f32le(&path, 4).expect("full row");
        assert_eq!(got, values);

        let short = dir.path().join("short.f32le");
        std::fs::write(&short, &values[0].to_le_bytes()).expect("short write");
        let err = read_retained_hidden_f32le(&short, 4).expect_err("short must raise");
        let msg = err.to_string();
        assert!(
            msg.contains("hidden size mismatch") || msg.contains("corrupt"),
            "unexpected error: {msg}"
        );

        let odd = dir.path().join("odd.f32le");
        std::fs::write(&odd, [0u8, 1, 2]).expect("odd write");
        let err = read_retained_hidden_f32le(&odd, 1).expect_err("odd must raise");
        assert!(err.to_string().contains("corrupt"));
    }

    #[test]
    fn emit_shape_is_doctor6_readable() {
        let g = CaptureGeometry::test_fixture();
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("run");
        let report = run_reduced_capture(&ReducedCaptureRequest {
            output_dir: out.clone(),
            geometry: g.clone(),
            set: CaptureSet::required(),
            probes: default_synthetic_probes(2, 8),
            max_per_expert: 3,
            row_threshold: 2,
            route_plan: RoutePlan::Cycling,
        })
        .expect("capture");
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(&report.result_path).unwrap()).unwrap();
        assert_eq!(doc["schema"], RESULT_SCHEMA);
        assert_eq!(doc["capture_summary"]["all_layer_activation_capture"], true);
        let probes = doc["probes"].as_array().unwrap();
        assert_eq!(probes.len(), 2);
        let step0 = &probes[0]["steps"][0];
        let layers = step0["layers"].as_array().unwrap();
        assert_eq!(layers.len(), g.layers);
        let row = &layers[0];
        assert!(row.get("layer").is_some());
        assert!(row.get("selected_expert_ids").is_some());
        assert!(row.get("normalized_route_weights").is_some());
        assert!(row.get("router_input_hidden_f32le").is_some());
        let hidden = &row["router_input_hidden_f32le"];
        if !hidden.is_null() {
            let rel = hidden["relative_path"].as_str().unwrap();
            assert!(rel.starts_with("hidden/L"));
            assert_eq!(hidden["elements"].as_u64().unwrap() as usize, g.hidden);
            let path = out.join(rel);
            let vals = read_retained_hidden_f32le(&path, g.hidden).expect("row");
            assert_eq!(vals.len(), g.hidden);
        }
        let n_fit = &doc["bounded_storage"]["n_fit_distribution"];
        assert!(n_fit.get("p10").is_some());
        assert!(n_fit.get("p50").is_some());
        assert!(n_fit.get("p90").is_some());
        assert!(n_fit.get("min").is_some());
        assert!(n_fit.get("max").is_some());
        assert!(n_fit.get("mean").is_some());
        assert!(n_fit.get("count_zero").is_some());
        assert!(n_fit.get("pct_zero").is_some());
        assert!(n_fit.get("frac_at_or_above_row_threshold").is_some());
        assert_eq!(
            n_fit["n_layer_expert_pairs"].as_u64().unwrap() as usize,
            g.layers * g.routed_experts
        );
        assert!(report.hidden_rows_retained_total > 0);
        let base = report.resident_after_append[0];
        for &r in &report.resident_after_append {
            assert_eq!(r, base);
        }
    }

    #[test]
    fn empty_write_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err =
            write_retained_hidden_f32le(&dir.path().join("empty.f32le"), &[]).expect_err("empty");
        assert!(err.to_string().contains("empty"));
    }
}
