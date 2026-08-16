//! Qwen3.8-27B (qwen3_5 text) geometry authority.
//!
//! Census: `receipts/ascent-2026-08-16/QWEN38_ARCH_CENSUS.json`.
//! Reuse: pack-time fuse of split `in_proj_{qkv,z,a,b}` into the Q80 QKVZ/BA
//! layout so existing Gated-DeltaNet kernels apply. This module does not
//! open Metal, pack weights, or generate tokens.

use crate::{Error, Result};
use serde_json::Value;
use std::path::Path;

pub const QWEN38_MODEL_ID: &str = "Qwen3.8-27B";
pub const QWEN38_SOURCE_REPOSITORY: &str = "PocketAiHub/Qwen3.8-27B-Abliterated-MLX";
pub const QWEN38_BASE_REPOSITORY: &str = "Qwen/Qwen3.8-27B";
pub const QWEN38_BASE_REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
pub const QWEN38_ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
pub const QWEN38_MODEL_TYPE: &str = "qwen3_5";
pub const QWEN38_TEXT_MODEL_TYPE: &str = "qwen3_5_text";

pub const QWEN38_LAYERS: usize = 64;
pub const QWEN38_DELTANET_LAYERS: usize = 48;
pub const QWEN38_GQA_LAYERS: usize = 16;
pub const QWEN38_FULL_ATTENTION_INTERVAL: usize = 4;
pub const QWEN38_HIDDEN: usize = 5_120;
pub const QWEN38_INTERMEDIATE: usize = 17_408;
pub const QWEN38_VOCAB: usize = 248_320;
pub const QWEN38_RMS_EPS: f32 = 1.0e-6;
pub const QWEN38_ROPE_THETA: f32 = 10_000_000.0;
pub const QWEN38_PARTIAL_ROTARY_FACTOR: f32 = 0.25;

pub const QWEN38_LINEAR_KEY_HEADS: usize = 16;
pub const QWEN38_LINEAR_VALUE_HEADS: usize = 48;
pub const QWEN38_LINEAR_VALUES_PER_KEY: usize = 3;
pub const QWEN38_LINEAR_KEY_HEAD_DIM: usize = 128;
pub const QWEN38_LINEAR_VALUE_HEAD_DIM: usize = 128;
pub const QWEN38_LINEAR_CONV_KERNEL: usize = 4;

pub const QWEN38_GQA_HEADS: usize = 24;
pub const QWEN38_GQA_KV_HEADS: usize = 4;
pub const QWEN38_GQA_HEAD_DIM: usize = 256;
pub const QWEN38_GQA_ROTARY_DIM: usize = 64;

pub const QWEN38_IN_PROJ_QKV_ROWS: usize = 10_240;
pub const QWEN38_IN_PROJ_Z_ROWS: usize = 6_144;
pub const QWEN38_IN_PROJ_A_ROWS: usize = 48;
pub const QWEN38_IN_PROJ_B_ROWS: usize = 48;
pub const QWEN38_QKVZ_ROWS: usize = 16_384;
pub const QWEN38_BA_ROWS: usize = 96;
pub const QWEN38_Q_PROJ_ROWS: usize = 12_288;
pub const QWEN38_KV_PROJ_ROWS: usize = 1_024;
pub const QWEN38_O_PROJ_ROWS: usize = 5_120;
pub const QWEN38_O_PROJ_COLS: usize = 6_144;

pub const QWEN38_LANGUAGE_PREFIX: &str = "language_model.";
pub const QWEN38_VISION_PREFIX: &str = "vision_tower.";

pub const QWEN38_EOS_IM_END: u32 = 248_046;
pub const QWEN38_EOS_END_OF_TEXT: u32 = 248_044;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen38MixerKind {
    DeltaNet,
    Gqa,
}

impl Qwen38MixerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DeltaNet => "delta_net",
            Self::Gqa => "gqa",
        }
    }

    pub const fn as_source_layer_type(self) -> &'static str {
        match self {
            Self::DeltaNet => "linear_attention",
            Self::Gqa => "full_attention",
        }
    }
}

/// Source rule: GQA iff `(layer + 1) % 4 == 0` (layers 3, 7, …, 63).
pub fn qwen38_mixer_kind(layer: usize) -> Result<Qwen38MixerKind> {
    if layer >= QWEN38_LAYERS {
        return Err(Error::Model(format!(
            "qwen38 layer {layer} is outside 0..{QWEN38_LAYERS}"
        )));
    }
    if (layer + 1) % QWEN38_FULL_ATTENTION_INTERVAL == 0 {
        Ok(Qwen38MixerKind::Gqa)
    } else {
        Ok(Qwen38MixerKind::DeltaNet)
    }
}

/// `slot = layer - layer / 4` (48 exclusive slots).
pub fn qwen38_deltanet_state_slot(layer: usize) -> Result<usize> {
    match qwen38_mixer_kind(layer)? {
        Qwen38MixerKind::DeltaNet => Ok(layer - layer / QWEN38_FULL_ATTENTION_INTERVAL),
        Qwen38MixerKind::Gqa => Err(Error::Model(format!(
            "qwen38 layer {layer} is GQA; DeltaNet slot is undefined"
        ))),
    }
}

/// `slot = layer / 4` (16 exclusive slots).
pub fn qwen38_gqa_state_slot(layer: usize) -> Result<usize> {
    match qwen38_mixer_kind(layer)? {
        Qwen38MixerKind::Gqa => Ok(layer / QWEN38_FULL_ATTENTION_INTERVAL),
        Qwen38MixerKind::DeltaNet => Err(Error::Model(format!(
            "qwen38 layer {layer} is DeltaNet; GQA slot is undefined"
        ))),
    }
}

pub fn qwen38_layer_name(layer: usize, suffix: &str) -> String {
    format!("language_model.model.layers.{layer}.{suffix}")
}

pub fn qwen38_embed_name() -> &'static str {
    "language_model.model.embed_tokens.weight"
}

pub fn qwen38_lm_head_name() -> &'static str {
    "language_model.lm_head.weight"
}

pub fn qwen38_final_norm_name() -> &'static str {
    "language_model.model.norm.weight"
}

/// Accept `qwen3_5` / `qwen3_5_text`. Refuse Q30/Q80/MoE identities.
pub fn qwen38_accept_config(config: &Value) -> Result<Qwen38AcceptedConfig> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("");
    match model_type {
        "qwen3_5" | "qwen3_5_text" => {}
        "qwen3_next" | "qwen3_moe" | "qwen3" | "qwen2" | "qwen2_moe" => {
            return Err(Error::Model(format!(
                "qwen38 refuses model_type {model_type:?}; expected qwen3_5 / qwen3_5_text"
            )));
        }
        other if !other.is_empty() => {
            return Err(Error::Model(format!(
                "qwen38 refuses unknown model_type {other:?}"
            )));
        }
        _ => {
            return Err(Error::Model(
                "qwen38 config is missing model_type".into(),
            ));
        }
    }
    let text = config.get("text_config").unwrap_or(config);
    let text_type = text
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or(model_type);
    if text_type != "qwen3_5" && text_type != "qwen3_5_text" {
        return Err(Error::Model(format!(
            "qwen38 text_config.model_type {text_type:?} is not qwen3_5_text"
        )));
    }
    if text.get("num_experts").is_some() || text.get("moe_intermediate_size").is_some() {
        return Err(Error::Model(
            "qwen38 is dense; config contains MoE keys".into(),
        ));
    }
    let layers = required_usize(text, "num_hidden_layers")?;
    let hidden = required_usize(text, "hidden_size")?;
    let intermediate = required_usize(text, "intermediate_size")?;
    let vocab = required_usize(text, "vocab_size")?;
    let heads = required_usize(text, "num_attention_heads")?;
    let kv_heads = required_usize(text, "num_key_value_heads")?;
    let head_dim = required_usize(text, "head_dim")?;
    let interval = required_usize(text, "full_attention_interval")?;
    if layers != QWEN38_LAYERS
        || hidden != QWEN38_HIDDEN
        || intermediate != QWEN38_INTERMEDIATE
        || vocab != QWEN38_VOCAB
        || heads != QWEN38_GQA_HEADS
        || kv_heads != QWEN38_GQA_KV_HEADS
        || head_dim != QWEN38_GQA_HEAD_DIM
        || interval != QWEN38_FULL_ATTENTION_INTERVAL
    {
        return Err(Error::Model(format!(
            "qwen38 geometry drifted: layers={layers} hidden={hidden} intermediate={intermediate} vocab={vocab} heads={heads} kv={kv_heads} head_dim={head_dim} interval={interval}"
        )));
    }
    Ok(Qwen38AcceptedConfig {
        model_type: model_type.to_owned(),
        text_model_type: text_type.to_owned(),
        layers,
        hidden,
        intermediate,
        vocab,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38AcceptedConfig {
    pub model_type: String,
    pub text_model_type: String,
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
}

pub fn qwen38_load_and_accept_config(path: impl AsRef<Path>) -> Result<Qwen38AcceptedConfig> {
    let raw = std::fs::read(path.as_ref()).map_err(|error| {
        Error::Model(format!(
            "cannot read qwen38 config {}: {error}",
            path.as_ref().display()
        ))
    })?;
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|error| Error::Model(format!("qwen38 config is not JSON: {error}")))?;
    qwen38_accept_config(&value)
}

fn required_usize(object: &Value, key: &str) -> Result<usize> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Model(format!("qwen38 config missing usize {key}")))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen38DeltaNetLayout {
    pub hidden: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub values_per_key: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub qkvz_rows_per_key: usize,
    pub ba_rows_per_key: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}

impl Qwen38DeltaNetLayout {
    pub const fn source_exact() -> Self {
        let value_rows = QWEN38_LINEAR_VALUES_PER_KEY * QWEN38_LINEAR_VALUE_HEAD_DIM;
        Self {
            hidden: QWEN38_HIDDEN,
            key_heads: QWEN38_LINEAR_KEY_HEADS,
            value_heads: QWEN38_LINEAR_VALUE_HEADS,
            values_per_key: QWEN38_LINEAR_VALUES_PER_KEY,
            key_head_dim: QWEN38_LINEAR_KEY_HEAD_DIM,
            value_head_dim: QWEN38_LINEAR_VALUE_HEAD_DIM,
            qkvz_rows_per_key: QWEN38_LINEAR_KEY_HEAD_DIM * 2 + value_rows * 2,
            ba_rows_per_key: QWEN38_LINEAR_VALUES_PER_KEY * 2,
            conv_channels: QWEN38_LINEAR_KEY_HEADS * QWEN38_LINEAR_KEY_HEAD_DIM * 2
                + QWEN38_LINEAR_VALUE_HEADS * QWEN38_LINEAR_VALUE_HEAD_DIM,
            conv_kernel: QWEN38_LINEAR_CONV_KERNEL,
        }
    }

    pub fn qkvz_rows(&self) -> usize {
        self.key_heads * self.qkvz_rows_per_key
    }

    pub fn ba_rows(&self) -> usize {
        self.key_heads * self.ba_rows_per_key
    }

    pub fn value_elements(&self) -> usize {
        self.value_heads * self.value_head_dim
    }

    pub fn key_elements(&self) -> usize {
        self.key_heads * self.key_head_dim
    }

    pub fn conv_state_elements(&self) -> usize {
        self.conv_channels * (self.conv_kernel - 1)
    }

    pub fn recurrent_state_elements(&self) -> usize {
        self.value_heads * self.key_head_dim * self.value_head_dim
    }
}

/// Interleave split Q38 `in_proj_qkv` + `in_proj_z` into Q80 per-key-head
/// `Q128,K128,V384,Z384` rows. Weights are `[rows, hidden]` row-major.
pub fn fuse_in_proj_qkvz(qkv: &[f32], z: &[f32], hidden: usize) -> Result<Vec<f32>> {
    let layout = Qwen38DeltaNetLayout::source_exact();
    if hidden != layout.hidden {
        return Err(Error::Model(format!(
            "qwen38 QKVZ fuse hidden {hidden} != {}",
            layout.hidden
        )));
    }
    let qkv_rows = layout.key_elements() * 2 + layout.value_elements();
    let z_rows = layout.value_elements();
    if qkv.len() != qkv_rows * hidden {
        return Err(Error::Model(format!(
            "qwen38 in_proj_qkv has {} values, expected {}",
            qkv.len(),
            qkv_rows * hidden
        )));
    }
    if z.len() != z_rows * hidden {
        return Err(Error::Model(format!(
            "qwen38 in_proj_z has {} values, expected {}",
            z.len(),
            z_rows * hidden
        )));
    }
    let mut fused = vec![0.0f32; layout.qkvz_rows() * hidden];
    let value_rows = layout.values_per_key * layout.value_head_dim;
    for key_head in 0..layout.key_heads {
        let dst_base = key_head * layout.qkvz_rows_per_key;
        let q_src = key_head * layout.key_head_dim;
        let k_src = layout.key_elements() + key_head * layout.key_head_dim;
        let v_src = layout.key_elements() * 2 + key_head * value_rows;
        let z_src = key_head * value_rows;
        copy_rows(
            qkv,
            q_src,
            &mut fused,
            dst_base,
            layout.key_head_dim,
            hidden,
        );
        copy_rows(
            qkv,
            k_src,
            &mut fused,
            dst_base + layout.key_head_dim,
            layout.key_head_dim,
            hidden,
        );
        copy_rows(
            qkv,
            v_src,
            &mut fused,
            dst_base + layout.key_head_dim * 2,
            value_rows,
            hidden,
        );
        copy_rows(
            z,
            z_src,
            &mut fused,
            dst_base + layout.key_head_dim * 2 + value_rows,
            value_rows,
            hidden,
        );
    }
    Ok(fused)
}

/// Pack split `in_proj_b` + `in_proj_a` into `[key_head][b×3, a×3]`.
pub fn fuse_in_proj_ba(b: &[f32], a: &[f32], hidden: usize) -> Result<Vec<f32>> {
    let layout = Qwen38DeltaNetLayout::source_exact();
    if hidden != layout.hidden {
        return Err(Error::Model(format!(
            "qwen38 BA fuse hidden {hidden} != {}",
            layout.hidden
        )));
    }
    if a.len() != layout.value_heads * hidden || b.len() != layout.value_heads * hidden {
        return Err(Error::Model(format!(
            "qwen38 in_proj_a/b have {}/{} values, expected {}",
            a.len(),
            b.len(),
            layout.value_heads * hidden
        )));
    }
    let mut fused = vec![0.0f32; layout.ba_rows() * hidden];
    for key_head in 0..layout.key_heads {
        let src = key_head * layout.values_per_key;
        let dst = key_head * layout.ba_rows_per_key;
        copy_rows(b, src, &mut fused, dst, layout.values_per_key, hidden);
        copy_rows(
            a,
            src,
            &mut fused,
            dst + layout.values_per_key,
            layout.values_per_key,
            hidden,
        );
    }
    Ok(fused)
}

fn copy_rows(
    src: &[f32],
    src_row: usize,
    dst: &mut [f32],
    dst_row: usize,
    rows: usize,
    hidden: usize,
) {
    let src_off = src_row * hidden;
    let dst_off = dst_row * hidden;
    let n = rows * hidden;
    dst[dst_off..dst_off + n].copy_from_slice(&src[src_off..src_off + n]);
}

/// Inverse of [`fuse_in_proj_qkvz`] for the correctness gate.
pub fn split_fused_qkvz(fused: &[f32], hidden: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let layout = Qwen38DeltaNetLayout::source_exact();
    if fused.len() != layout.qkvz_rows() * hidden {
        return Err(Error::Model(format!(
            "qwen38 fused QKVZ has {} values, expected {}",
            fused.len(),
            layout.qkvz_rows() * hidden
        )));
    }
    let qkv_rows = layout.key_elements() * 2 + layout.value_elements();
    let mut qkv = vec![0.0f32; qkv_rows * hidden];
    let mut z = vec![0.0f32; layout.value_elements() * hidden];
    let value_rows = layout.values_per_key * layout.value_head_dim;
    for key_head in 0..layout.key_heads {
        let src_base = key_head * layout.qkvz_rows_per_key;
        let q_dst = key_head * layout.key_head_dim;
        let k_dst = layout.key_elements() + key_head * layout.key_head_dim;
        let v_dst = layout.key_elements() * 2 + key_head * value_rows;
        let z_dst = key_head * value_rows;
        copy_rows(
            fused,
            src_base,
            &mut qkv,
            q_dst,
            layout.key_head_dim,
            hidden,
        );
        copy_rows(
            fused,
            src_base + layout.key_head_dim,
            &mut qkv,
            k_dst,
            layout.key_head_dim,
            hidden,
        );
        copy_rows(
            fused,
            src_base + layout.key_head_dim * 2,
            &mut qkv,
            v_dst,
            value_rows,
            hidden,
        );
        copy_rows(
            fused,
            src_base + layout.key_head_dim * 2 + value_rows,
            &mut z,
            z_dst,
            value_rows,
            hidden,
        );
    }
    Ok((qkv, z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mixer_is_48_deltanet_16_gqa_interval_4() {
        let mut dn = 0;
        let mut gqa = 0;
        for layer in 0..QWEN38_LAYERS {
            match qwen38_mixer_kind(layer).unwrap() {
                Qwen38MixerKind::DeltaNet => {
                    dn += 1;
                    assert_eq!(
                        qwen38_deltanet_state_slot(layer).unwrap(),
                        layer - layer / 4
                    );
                }
                Qwen38MixerKind::Gqa => {
                    gqa += 1;
                    assert_eq!(qwen38_gqa_state_slot(layer).unwrap(), layer / 4);
                    assert_eq!((layer + 1) % 4, 0);
                }
            }
        }
        assert_eq!(dn, QWEN38_DELTANET_LAYERS);
        assert_eq!(gqa, QWEN38_GQA_LAYERS);
        assert_eq!(qwen38_deltanet_state_slot(0).unwrap(), 0);
        assert_eq!(qwen38_deltanet_state_slot(62).unwrap(), 47);
        assert_eq!(qwen38_gqa_state_slot(3).unwrap(), 0);
        assert_eq!(qwen38_gqa_state_slot(63).unwrap(), 15);
    }

    #[test]
    fn layout_matches_census() {
        let layout = Qwen38DeltaNetLayout::source_exact();
        assert_eq!(layout.qkvz_rows(), QWEN38_QKVZ_ROWS);
        assert_eq!(layout.ba_rows(), QWEN38_BA_ROWS);
        assert_eq!(layout.conv_channels, QWEN38_IN_PROJ_QKV_ROWS);
        assert_eq!(layout.value_elements(), 6_144);
        assert_eq!(layout.recurrent_state_elements(), 48 * 128 * 128);
        assert_eq!(layout.conv_state_elements(), 10_240 * 3);
    }

    #[test]
    fn qkvz_and_ba_fuse_are_invertible() {
        let hidden = QWEN38_HIDDEN;
        let qkv_rows = QWEN38_IN_PROJ_QKV_ROWS;
        let z_rows = QWEN38_IN_PROJ_Z_ROWS;
        let mut qkv = vec![0.0f32; qkv_rows * hidden];
        let mut z = vec![0.0f32; z_rows * hidden];
        for i in 0..qkv.len() {
            qkv[i] = (i as f32) * 0.001 + 0.25;
        }
        for i in 0..z.len() {
            z[i] = -(i as f32) * 0.002 + 1.5;
        }
        let fused = fuse_in_proj_qkvz(&qkv, &z, hidden).unwrap();
        let (qkv2, z2) = split_fused_qkvz(&fused, hidden).unwrap();
        assert_eq!(qkv, qkv2);
        assert_eq!(z, z2);

        let mut a = vec![0.0f32; 48 * hidden];
        let mut b = vec![0.0f32; 48 * hidden];
        for i in 0..a.len() {
            a[i] = i as f32 * 0.01;
            b[i] = -(i as f32) * 0.02;
        }
        let ba = fuse_in_proj_ba(&b, &a, hidden).unwrap();
        assert_eq!(ba.len(), QWEN38_BA_ROWS * hidden);
        // key head 1: b rows 3..6 then a rows 3..6
        let dst = 1 * 6 * hidden;
        assert_eq!(&ba[dst..dst + hidden], &b[3 * hidden..4 * hidden]);
        assert_eq!(
            &ba[dst + 3 * hidden..dst + 4 * hidden],
            &a[3 * hidden..4 * hidden]
        );
    }

    #[test]
    fn config_accepts_qwen3_5_and_refuses_q80() {
        let ok = json!({
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 248320,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "full_attention_interval": 4
            }
        });
        qwen38_accept_config(&ok).unwrap();
        let bad = json!({
            "model_type": "qwen3_next",
            "text_config": {
                "model_type": "qwen3_next",
                "num_hidden_layers": 48,
                "hidden_size": 2048,
                "intermediate_size": 512,
                "vocab_size": 151936,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "full_attention_interval": 4
            }
        });
        assert!(qwen38_accept_config(&bad).is_err());
    }
}
