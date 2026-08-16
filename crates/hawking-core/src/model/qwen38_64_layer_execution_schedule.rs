//! Qwen3.8 64-layer execution schedule.
//!
//! Same interval-4 mixer rule as Q80. 48 DeltaNet slots + 16 GQA slots.
//! Dense SwiGLU suffix replaces Q80's 14-dispatch MoE suffix.

use super::qwen38_geometry::{
    qwen38_deltanet_state_slot, qwen38_gqa_state_slot, qwen38_mixer_kind, Qwen38MixerKind,
    QWEN38_DELTANET_LAYERS, QWEN38_GQA_LAYERS, QWEN38_HIDDEN, QWEN38_LAYERS,
};
use crate::{Error, Result};

pub const QWEN38_MIXER_PREFIX_DISPATCHES: usize = 9;
pub const QWEN38_DENSE_MLP_SUFFIX_DISPATCHES: usize = 6;
pub const QWEN38_FULL_LAYER_DISPATCHES: usize =
    QWEN38_MIXER_PREFIX_DISPATCHES + QWEN38_DENSE_MLP_SUFFIX_DISPATCHES;

pub const QWEN38_DELTANET_MIXER_PREFIX_KERNELS: [&str; QWEN38_MIXER_PREFIX_DISPATCHES] = [
    "qwen80_residual_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen38_qkvz_rearrange_conv_l2_f32",
    "qwen80_ba_to_decay_beta_f32",
    "qwen38_gated_delta_decode_vi",
    "qwen80_deltanet_gated_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_next_add_residual",
];

pub const QWEN38_GQA_MIXER_PREFIX_KERNELS: [&str; QWEN38_MIXER_PREFIX_DISPATCHES] = [
    "qwen80_residual_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen38_gqa_qk_norm_rope_cache_f32",
    "mha_decode_f32",
    "qwen38_attention_apply_sigmoid_gate",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_next_add_residual",
];

pub const QWEN38_DENSE_MLP_SUFFIX_KERNELS: [&str; QWEN38_DENSE_MLP_SUFFIX_DISPATCHES] = [
    "qwen80_residual_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen80_silu_mul_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "qwen_next_add_residual",
];

pub const QWEN38_TERMINAL_HEAD_KERNELS: [&str; 3] = [
    "qwen80_residual_rmsnorm_f32",
    "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
    "sample_argmax_f32",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen38LayerExecutionSchedule {
    pub layer: usize,
    pub mixer: Qwen38MixerKind,
    pub source_layer_type: &'static str,
    pub state_slot: usize,
    pub mixer_prefix_kernel_names: &'static [&'static str],
    pub mlp_suffix_kernel_names: &'static [&'static str],
    pub full_layer_dispatch_count: usize,
}

pub fn qwen38_layer_execution_schedule(layer: usize) -> Result<Qwen38LayerExecutionSchedule> {
    let mixer = qwen38_mixer_kind(layer)?;
    let (slot, prefix) = match mixer {
        Qwen38MixerKind::DeltaNet => (
            qwen38_deltanet_state_slot(layer)?,
            QWEN38_DELTANET_MIXER_PREFIX_KERNELS.as_slice(),
        ),
        Qwen38MixerKind::Gqa => (
            qwen38_gqa_state_slot(layer)?,
            QWEN38_GQA_MIXER_PREFIX_KERNELS.as_slice(),
        ),
    };
    Ok(Qwen38LayerExecutionSchedule {
        layer,
        mixer,
        source_layer_type: mixer.as_source_layer_type(),
        state_slot: slot,
        mixer_prefix_kernel_names: prefix,
        mlp_suffix_kernel_names: QWEN38_DENSE_MLP_SUFFIX_KERNELS.as_slice(),
        full_layer_dispatch_count: QWEN38_FULL_LAYER_DISPATCHES,
    })
}

pub fn qwen38_all_64_layer_execution_schedules() -> Result<Vec<Qwen38LayerExecutionSchedule>> {
    (0..QWEN38_LAYERS).map(qwen38_layer_execution_schedule).collect()
}

pub fn qwen38_assert_schedule_intact() -> Result<()> {
    let layers = qwen38_all_64_layer_execution_schedules()?;
    if layers.len() != QWEN38_LAYERS {
        return Err(Error::Model(format!(
            "qwen38 schedule length {} != {QWEN38_LAYERS}",
            layers.len()
        )));
    }
    let mut dn = 0usize;
    let mut gqa = 0usize;
    let mut dn_slots = Vec::new();
    let mut gqa_slots = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        if layer.layer != index {
            return Err(Error::Model(format!(
                "qwen38 schedule[{index}].layer={}",
                layer.layer
            )));
        }
        if layer.full_layer_dispatch_count != QWEN38_FULL_LAYER_DISPATCHES {
            return Err(Error::Model(format!(
                "qwen38 layer {index} dispatch count drifted"
            )));
        }
        if layer.mlp_suffix_kernel_names != QWEN38_DENSE_MLP_SUFFIX_KERNELS {
            return Err(Error::Model(format!(
                "qwen38 layer {index} MLP suffix drifted"
            )));
        }
        match layer.mixer {
            Qwen38MixerKind::DeltaNet => {
                dn += 1;
                dn_slots.push(layer.state_slot);
                if layer.mixer_prefix_kernel_names != QWEN38_DELTANET_MIXER_PREFIX_KERNELS {
                    return Err(Error::Model(format!(
                        "qwen38 layer {index} DeltaNet prefix drifted"
                    )));
                }
            }
            Qwen38MixerKind::Gqa => {
                gqa += 1;
                gqa_slots.push(layer.state_slot);
                if layer.mixer_prefix_kernel_names != QWEN38_GQA_MIXER_PREFIX_KERNELS {
                    return Err(Error::Model(format!(
                        "qwen38 layer {index} GQA prefix drifted"
                    )));
                }
            }
        }
    }
    if dn != QWEN38_DELTANET_LAYERS || gqa != QWEN38_GQA_LAYERS {
        return Err(Error::Model(format!(
            "qwen38 mixer counts dn={dn} gqa={gqa}"
        )));
    }
    dn_slots.sort_unstable();
    gqa_slots.sort_unstable();
    if dn_slots != (0..QWEN38_DELTANET_LAYERS).collect::<Vec<_>>() {
        return Err(Error::Model(format!(
            "qwen38 DeltaNet slots are not 0..{QWEN38_DELTANET_LAYERS}: {dn_slots:?}"
        )));
    }
    if gqa_slots != (0..QWEN38_GQA_LAYERS).collect::<Vec<_>>() {
        return Err(Error::Model(format!(
            "qwen38 GQA slots are not 0..{QWEN38_GQA_LAYERS}: {gqa_slots:?}"
        )));
    }
    if QWEN38_HIDDEN != 5_120 {
        return Err(Error::Model("qwen38 hidden drifted".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_64_layers_dense_suffix_no_moe() {
        qwen38_assert_schedule_intact().unwrap();
        let layer0 = qwen38_layer_execution_schedule(0).unwrap();
        assert_eq!(layer0.mixer, Qwen38MixerKind::DeltaNet);
        assert!(!layer0
            .mlp_suffix_kernel_names
            .iter()
            .any(|name| name.contains("expert") || name.contains("router") || name.contains("moe")));
        let layer3 = qwen38_layer_execution_schedule(3).unwrap();
        assert_eq!(layer3.mixer, Qwen38MixerKind::Gqa);
        assert_eq!(layer3.state_slot, 0);
        assert_eq!(
            qwen38_layer_execution_schedule(63).unwrap().state_slot,
            15
        );
    }
}
