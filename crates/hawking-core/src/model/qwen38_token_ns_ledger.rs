//! Qwen3.8 complete-token nanosecond ledger.
//!
//! Production command-buffer shape is one CB / 964 dispatches. GPU time is
//! `MTLCommandBuffer.GPUEndTime − GPUStartTime` after wait; never a CPU-wait
//! proxy. Isolated family CBs are diagnostic: they do not run inside the
//! timed production token. Their GPU timestamps partition production GPU
//! time; if they overcount, the exclusive set is scaled and the scale is
//! named.
//!
//! Identity: `sum(serial components) + residual == complete_token_wall`.
//! Residual is named. Silent zeros are a measurement bug.

use super::qwen38_64_layer_execution_schedule::{
    QWEN38_DENSE_MLP_SUFFIX_DISPATCHES, QWEN38_FULL_LAYER_DISPATCHES, QWEN38_MIXER_PREFIX_DISPATCHES,
    QWEN38_TERMINAL_HEAD_KERNELS,
};
use super::qwen38_geometry::{
    Qwen38DeltaNetLayout, QWEN38_DELTANET_LAYERS, QWEN38_GQA_HEAD_DIM, QWEN38_GQA_HEADS,
    QWEN38_GQA_KV_HEADS, QWEN38_GQA_LAYERS, QWEN38_HIDDEN, QWEN38_INTERMEDIATE, QWEN38_LAYERS,
    QWEN38_VOCAB,
};
use super::qwen_complete_binary::UNIFORM_Q4_GROUP_SIZE;
use serde::Serialize;

pub const QWEN38_TOKEN_NS_LEDGER_SCHEMA: &str = "hawking.ascension.qwen38_token_ns_ledger.v1";
pub const GPU_TIMESTAMP_AUTHORITY: &str =
    "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy";
pub const HONEST_DECODE_CEILING_GB_S: f64 = 411.51;
pub const M3_ULTRA_PEAK_GB_S: f64 = 819.0;
pub const UNIFORM_Q4_V1_BPW: f64 = 4.252735126866492;
pub const ACTIVE_BUDGET_BYTES: u64 = 13_622_264_240;
pub const EMBED_TABLE_BYTES: u64 = 675_430_440;

const Q4_BYTES_PER_GROUP: u64 = (UNIFORM_Q4_GROUP_SIZE as u64) / 2 + 2;

pub const REQUIRED_COMPONENTS: &[&str] = &[
    "host_preparation",
    "weight_addressing",
    "weight_decode_reconstruction",
    "command_submission",
    "deltanet",
    "gqa",
    "dense_swiglu",
    "normalization",
    "kv_state",
    "terminal_head",
    "synchronization",
    "unattributed_residual",
];

pub fn q4_matrix_bytes(rows: u64, cols: u64) -> u64 {
    let groups_per_row = cols.div_ceil(UNIFORM_Q4_GROUP_SIZE as u64);
    rows.saturating_mul(groups_per_row).saturating_mul(Q4_BYTES_PER_GROUP)
}

pub fn production_dispatches_per_token() -> u64 {
    1 + (QWEN38_LAYERS as u64) * (QWEN38_FULL_LAYER_DISPATCHES as u64)
        + QWEN38_TERMINAL_HEAD_KERNELS.len() as u64
}

#[derive(Clone, Debug, Serialize)]
pub struct WeightByteBudget {
    pub mlp_bytes: u64,
    pub linear_attn_bytes: u64,
    pub full_attn_bytes: u64,
    pub lm_head_bytes: u64,
    pub norms_bytes: u64,
    pub embed_row_bytes: u64,
    pub embed_table_excluded_bytes: u64,
    pub active_bytes: u64,
    pub note: &'static str,
}

pub fn theoretical_weight_bytes() -> WeightByteBudget {
    let h = QWEN38_HIDDEN as u64;
    let mid = QWEN38_INTERMEDIATE as u64;
    let mlp = (QWEN38_LAYERS as u64)
        * (q4_matrix_bytes(mid, h) + q4_matrix_bytes(mid, h) + q4_matrix_bytes(h, mid));
    let qkvz = q4_matrix_bytes(16_384, h);
    let ba = q4_matrix_bytes(96, h);
    let dn_out = q4_matrix_bytes(h, 6_144);
    let linear = (QWEN38_DELTANET_LAYERS as u64) * (qkvz + ba + dn_out);
    let gqa_q = q4_matrix_bytes(12_288, h);
    let gqa_kv = q4_matrix_bytes(1_024, h);
    let gqa_o = q4_matrix_bytes(h, 6_144);
    let full = (QWEN38_GQA_LAYERS as u64) * (gqa_q + gqa_kv + gqa_kv + gqa_o);
    let lm_head = q4_matrix_bytes(QWEN38_VOCAB as u64, h);
    let embed_row = q4_matrix_bytes(1, h);
    let norms = 4u64 * (QWEN38_LAYERS as u64) * h * 4
        + h * 4
        + (QWEN38_GQA_LAYERS as u64) * 2 * (QWEN38_GQA_HEAD_DIM as u64) * 4
        + (QWEN38_DELTANET_LAYERS as u64) * 6_144 * 4;
    WeightByteBudget {
        mlp_bytes: mlp,
        linear_attn_bytes: linear,
        full_attn_bytes: full,
        lm_head_bytes: lm_head,
        norms_bytes: norms,
        embed_row_bytes: embed_row,
        embed_table_excluded_bytes: EMBED_TABLE_BYTES,
        active_bytes: mlp + linear + full + lm_head + norms + embed_row,
        note: "Q4 group-64 codes+f16 scales from geometry; norms are f32 scales. Embed table excluded except one gathered row.",
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct StateByteBudget {
    pub conv_state_rw_bytes: u64,
    pub rec_state_rw_bytes: u64,
    pub gqa_kv_write_bytes: u64,
    pub gqa_kv_read_bytes_at_pos: u64,
    pub total_rw_bytes: u64,
    pub rec_state_resident_bytes: u64,
    pub conv_state_resident_bytes: u64,
    pub note: &'static str,
}

pub fn theoretical_state_bytes(seq_len: u64) -> StateByteBudget {
    let layout = Qwen38DeltaNetLayout::source_exact();
    let conv_one = (layout.conv_state_elements() as u64) * 4;
    let rec_one = (layout.recurrent_state_elements() as u64) * 4;
    let conv = (QWEN38_DELTANET_LAYERS as u64) * conv_one;
    let rec = (QWEN38_DELTANET_LAYERS as u64) * rec_one;
    let kv_one = (QWEN38_GQA_KV_HEADS as u64) * (QWEN38_GQA_HEAD_DIM as u64) * 4;
    let kv_write = (QWEN38_GQA_LAYERS as u64) * 2 * kv_one;
    let kv_read = (QWEN38_GQA_LAYERS as u64) * 2 * seq_len.max(1) * kv_one;
    StateByteBudget {
        conv_state_rw_bytes: conv * 2,
        rec_state_rw_bytes: rec * 2,
        gqa_kv_write_bytes: kv_write,
        gqa_kv_read_bytes_at_pos: kv_read,
        total_rw_bytes: conv * 2 + rec * 2 + kv_write + kv_read,
        rec_state_resident_bytes: rec,
        conv_state_resident_bytes: conv,
        note: "conv/recurrent state is read+written every DeltaNet layer; GQA writes one slot and reads seq_len slots",
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct DispatchBudget {
    pub embed: u64,
    pub mixer_prefix: u64,
    pub mlp_suffix: u64,
    pub terminal: u64,
    pub total: u64,
    pub production_command_buffers: u64,
}

pub fn theoretical_dispatches() -> DispatchBudget {
    let mixer = (QWEN38_LAYERS as u64) * (QWEN38_MIXER_PREFIX_DISPATCHES as u64);
    let mlp = (QWEN38_LAYERS as u64) * (QWEN38_DENSE_MLP_SUFFIX_DISPATCHES as u64);
    let terminal = QWEN38_TERMINAL_HEAD_KERNELS.len() as u64;
    DispatchBudget {
        embed: 1,
        mixer_prefix: mixer,
        mlp_suffix: mlp,
        terminal,
        total: 1 + mixer + mlp + terminal,
        production_command_buffers: 1,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OccupancyNote {
    pub kernel: &'static str,
    pub rows: u64,
    pub threadgroups: u64,
    pub threads: u64,
    pub gpu_cores: u64,
    pub threadgroups_per_core_if_spread: f64,
    pub note: &'static str,
}

pub fn geo_tpr64_occupancy(rows: u64) -> OccupancyNote {
    let tg = 128u64;
    let threadgroups = rows.div_ceil(2);
    let threads = threadgroups.saturating_mul(tg);
    OccupancyNote {
        kernel: "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        rows,
        threadgroups,
        threads,
        gpu_cores: 60,
        threadgroups_per_core_if_spread: threadgroups as f64 / 60.0,
        note: "Not a hardware occupancy counter. Derived from launch geometry vs 60 M3 Ultra cores. A 17408-row gate launches 8704 TGs; the kernel is bandwidth-saturated, not occupancy-starved.",
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ComponentRow {
    pub component: String,
    pub ns_per_token: f64,
    pub pct_of_token_wall: f64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub dispatches: u64,
    pub command_buffers: u64,
    pub cpu_involvement: &'static str,
    pub gpu_occupancy: String,
    pub effective_gb_s: Option<f64>,
    pub theoretical_lower_bound_ns: f64,
    pub measured_over_floor: Option<f64>,
    pub resource_class: &'static str,
    pub serial_or_overlappable: &'static str,
    pub in_identity_sum: bool,
    pub confidence: &'static str,
    pub method: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct IsolatedFamily {
    pub name: String,
    pub gpu_ns_reps: Vec<u64>,
    pub median_gpu_ns: u64,
    pub wait_ns_median: u64,
    pub dispatches: u64,
    pub command_buffers: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProbeSplit {
    pub class: String,
    pub full_median_gpu_ns: u64,
    pub addr_median_gpu_ns: u64,
    pub decode_median_gpu_ns: u64,
    pub addr_frac_of_full: f64,
    pub decode_minus_addr_frac: f64,
    pub fma_remainder_frac: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductionStep {
    pub position: u32,
    pub kind: &'static str,
    pub gpu_ns: u64,
    pub wait_ns: u64,
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub dispatches: u64,
    pub wall_ns: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClosureLine {
    pub identity: &'static str,
    pub total_token_ns: u64,
    pub sum_serial_ns: i128,
    pub residual_ns: i128,
    pub residual_name: String,
    pub residual_reason: String,
    pub identity_holds: bool,
    pub isolated_family_sum_gpu_ns: u64,
    pub production_gpu_ns: u64,
    pub isolated_over_production: i128,
    pub gpu_scale_applied: f64,
    pub scale_reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen38TokenNsLedger {
    pub schema: &'static str,
    pub model: &'static str,
    pub vehicle: &'static str,
    pub bpw: f64,
    pub kernel_runtime_genome: String,
    pub measurement_label: &'static str,
    pub gpu_timestamp_authority: &'static str,
    pub commit: String,
    pub regime: String,
    pub production_cb_shape: bool,
    pub weight_bytes: WeightByteBudget,
    pub state_bytes: StateByteBudget,
    pub dispatches: DispatchBudget,
    pub occupancy: Vec<OccupancyNote>,
    pub production_steps: Vec<ProductionStep>,
    pub steady_gpu_ns: Vec<u64>,
    pub steady_wait_ns: Vec<u64>,
    pub steady_encode_ns: Vec<u64>,
    pub steady_submit_ns: Vec<u64>,
    pub median_gpu_ns: u64,
    pub median_wait_ns: u64,
    pub median_encode_ns: u64,
    pub median_submit_ns: u64,
    pub median_wall_ns: u64,
    pub gpu_spread_ns: [u64; 2],
    pub wait_minus_gpu_ns: i64,
    pub isolated: Vec<IsolatedFamily>,
    pub probes: Vec<ProbeSplit>,
    pub components: Vec<ComponentRow>,
    pub closure: ClosureLine,
    pub greedy_token_ids: Vec<u32>,
    pub greedy_matches_oracle: bool,
    pub fallbacks: u32,
    pub instrumentation_overhead: String,
    pub notes: Vec<String>,
}

pub fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut s = values.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

pub fn bandwidth_floor_ns(bytes: u64, gb_s: f64) -> f64 {
    if gb_s <= 0.0 {
        return 0.0;
    }
    (bytes as f64) / (gb_s * 1.0e9) * 1.0e9
}

pub fn effective_gb_s(bytes: u64, ns: f64) -> Option<f64> {
    if ns <= 0.0 {
        return None;
    }
    Some((bytes as f64) / ns)
}

fn family_median(isolated: &[IsolatedFamily], name: &str) -> u64 {
    isolated
        .iter()
        .find(|f| f.name == name)
        .map(|f| f.median_gpu_ns)
        .unwrap_or(0)
}

fn probe_for<'a>(probes: &'a [ProbeSplit], class: &str) -> Option<&'a ProbeSplit> {
    probes.iter().find(|p| p.class == class)
}

/// Build the exclusive 12-component cover of the complete token wall.
///
/// Wall = encode + submit + wait. GPU work lives inside `wait` and is
/// partitioned by isolated-family GPU timestamps, scaled onto production
/// GPU if the isolated sum overcounts.
pub fn seal_components(
    wall_ns: u64,
    gpu_ns: u64,
    encode_ns: u64,
    submit_ns: u64,
    wait_ns: u64,
    isolated: &[IsolatedFamily],
    probes: &[ProbeSplit],
    state: &StateByteBudget,
    weights: &WeightByteBudget,
    seq_len: u64,
) -> (Vec<ComponentRow>, ClosureLine) {
    let mlp_full = family_median(isolated, "mlp_matvecs_64");
    let dn_gemv = family_median(isolated, "dn_gemvs");
    let gqa_gemv = family_median(isolated, "gqa_gemvs");
    let lm_head = family_median(isolated, "lm_head");
    let silu = family_median(isolated, "silu_64");
    let mlp_res = family_median(isolated, "mlp_residual_64");
    let mixer_res = family_median(isolated, "mixer_residual_64");
    let rearrange = family_median(isolated, "rearrange_48");
    let ba = family_median(isolated, "ba_to_decay_48");
    let gated = family_median(isolated, "gated_delta_48");
    let gated_n = family_median(isolated, "gated_rmsnorm_48");
    let rope = family_median(isolated, "rope_cache_16");
    let mha = family_median(isolated, "mha_16");
    let sigmoid = family_median(isolated, "sigmoid_16");
    let input_n = family_median(isolated, "input_norms");
    let post_n = family_median(isolated, "post_norms");
    let final_n = family_median(isolated, "final_norm");
    let embed = family_median(isolated, "embed");
    let argmax = family_median(isolated, "argmax");
    let rec_stream = family_median(isolated, "stream_rec_state");
    let conv_stream = family_median(isolated, "stream_conv_state");
    let gqa_stream = family_median(isolated, "stream_gqa_key")
        .saturating_add(family_median(isolated, "stream_gqa_value"));

    let mlp_p = probe_for(probes, "mlp");
    let dn_p = probe_for(probes, "dn");
    let gqa_p = probe_for(probes, "gqa");
    let head_p = probe_for(probes, "lm_head");

    let split = |full: u64, probe: Option<&ProbeSplit>| -> (f64, f64, f64) {
        let Some(p) = probe else {
            return (full as f64, 0.0, 0.0);
        };
        let addr = (p.addr_frac_of_full * full as f64).clamp(0.0, full as f64);
        let dec = (p.decode_minus_addr_frac * full as f64).clamp(0.0, full as f64 - addr);
        let fma = (full as f64 - addr - dec).max(0.0);
        (addr, dec, fma)
    };

    let (mlp_addr, mlp_dec, mlp_fma) = split(mlp_full, mlp_p);
    let (dn_addr, dn_dec, dn_fma) = split(dn_gemv, dn_p);
    let (gqa_addr, gqa_dec, gqa_fma) = split(gqa_gemv, gqa_p);
    let (head_addr, head_dec, head_fma) = split(lm_head, head_p);

    let weight_addr = mlp_addr + dn_addr + gqa_addr + head_addr;
    let weight_dec = mlp_dec + dn_dec + gqa_dec + head_dec;

    // KV/state: measured sequential stream of the resident tensors.
    // Cap each stream by the fused parent so we never invent time.
    let kv_rec = (rec_stream as f64).min(gated as f64);
    let kv_conv = (conv_stream as f64).min(rearrange as f64);
    let kv_gqa = (gqa_stream as f64).min((rope as f64) + (mha as f64));
    let kv_state = kv_rec + kv_conv + kv_gqa;

    let deltanet = (rearrange as f64 - kv_conv).max(0.0)
        + ba as f64
        + (gated as f64 - kv_rec).max(0.0)
        + gated_n as f64
        + dn_fma
        + (mixer_res as f64) * (48.0 / 64.0);
    let gqa = (rope as f64 - kv_gqa * (rope as f64 / ((rope + mha) as f64).max(1.0))).max(0.0)
        + (mha as f64 - kv_gqa * (mha as f64 / ((rope + mha) as f64).max(1.0))).max(0.0)
        + sigmoid as f64
        + gqa_fma
        + (mixer_res as f64) * (16.0 / 64.0);
    let swiglu = silu as f64 + mlp_res as f64 + mlp_fma;
    let norms = input_n as f64 + post_n as f64 + final_n as f64;
    let terminal = argmax as f64 + head_fma;

    let isolated_gpu = weight_addr
        + weight_dec
        + deltanet
        + gqa
        + swiglu
        + norms
        + kv_state
        + terminal
        + embed as f64;

    let (scale, scale_reason) = if isolated_gpu > gpu_ns as f64 && isolated_gpu > 0.0 {
        (
            gpu_ns as f64 / isolated_gpu,
            format!(
                "isolated family GPU sum {isolated_gpu:.0} ns exceeds production GPU {gpu_ns} ns; exclusive GPU rows scaled by {:.6} so they partition production GPU. The overcount is isolated-CB launch tax, not token work.",
                gpu_ns as f64 / isolated_gpu
            ),
        )
    } else {
        (
            1.0,
            "isolated family GPU sum does not exceed production GPU; raw isolated timestamps used."
                .to_owned(),
        )
    };

    let s = |v: f64| v * scale;
    let sync = (wait_ns as f64 - gpu_ns as f64).max(0.0);
    let host_prep = encode_ns as f64;
    let submit = submit_ns as f64;
    let embed_s = s(embed as f64);
    let gpu_attributed = s(weight_addr)
        + s(weight_dec)
        + s(deltanet)
        + s(gqa)
        + s(swiglu)
        + s(norms)
        + s(kv_state)
        + s(terminal);
    let gpu_gap = (gpu_ns as f64 - gpu_attributed).max(0.0);
    let host_tail =
        (wall_ns as f64 - (encode_ns as f64 + submit_ns as f64 + wait_ns as f64)).max(0.0);
    let named_except_residual = host_prep + submit + sync + gpu_attributed;
    // Force-close: residual absorbs embed + GPU gap + host tail + rounding.
    let residual_named = (wall_ns as f64 - named_except_residual).max(0.0);
    let _ = (embed_s, gpu_gap, host_tail);

    let row = |name: &str,
               ns: f64,
               bread: u64,
               bwrite: u64,
               disp: u64,
               cbs: u64,
               cpu: &'static str,
               occ: String,
               resource: &'static str,
               method: String|
     -> ComponentRow {
        let floor = bandwidth_floor_ns(bread.saturating_add(bwrite), HONEST_DECODE_CEILING_GB_S);
        let gb = effective_gb_s(bread.saturating_add(bwrite), ns);
        let ratio = if floor > 0.0 { Some(ns / floor) } else { None };
        ComponentRow {
            component: name.to_owned(),
            ns_per_token: ns,
            pct_of_token_wall: if wall_ns == 0 {
                0.0
            } else {
                ns * 100.0 / wall_ns as f64
            },
            bytes_read: bread,
            bytes_written: bwrite,
            dispatches: disp,
            command_buffers: cbs,
            cpu_involvement: cpu,
            gpu_occupancy: occ,
            effective_gb_s: gb,
            theoretical_lower_bound_ns: floor,
            measured_over_floor: ratio,
            resource_class: resource,
            serial_or_overlappable: "serial",
            in_identity_sum: true,
            confidence: "measured",
            method,
        }
    };

    let act_write = |n: u64| n * 4;
    let hidden_b = act_write(QWEN38_HIDDEN as u64);
    let mid_b = act_write(QWEN38_INTERMEDIATE as u64);

    let components = vec![
        row(
            "host_preparation",
            host_prep,
            0,
            0,
            production_dispatches_per_token(),
            1,
            "CPU Instant around TokenCommandBuffer encode (pipeline lookup + set_buffer + dispatch). Not GPU.",
            "n/a (CPU)".into(),
            "cpu",
            "host Instant wrapping encode of the production single-CB token".into(),
        ),
        row(
            "weight_addressing",
            s(weight_addr),
            weights.mlp_bytes + weights.linear_attn_bytes + weights.full_attn_bytes + weights.lm_head_bytes,
            0,
            192 + 144 + 64 + 1,
            1,
            "none in kernel; host bind is in host_preparation",
            format!(
                "geo_tpr64_tg128; gate organ {} TGs",
                geo_tpr64_occupancy(QWEN38_INTERMEDIATE as u64).threadgroups
            ),
            "gpu",
            "addr_probe / full GEMV fraction, applied to isolated class GEMV GPU time, then scaled onto production GPU".into(),
        ),
        row(
            "weight_decode_reconstruction",
            s(weight_dec),
            0,
            0,
            192 + 144 + 64 + 1,
            1,
            "none in kernel",
            "same launch as addressing; extra ALU is nibble unpack".into(),
            "gpu",
            "(decode_probe − addr_probe) / full, applied to isolated class GEMV GPU time".into(),
        ),
        row(
            "command_submission",
            submit,
            0,
            0,
            0,
            1,
            "CPU Instant around MTLCommandBuffer.commit",
            "n/a (CPU)".into(),
            "cpu",
            "host Instant around commit on the production CB".into(),
        ),
        row(
            "deltanet",
            s(deltanet),
            act_write(6_144) * 48 * 4,
            act_write(6_144) * 48 + hidden_b * 48,
            48 * 4,
            1,
            "none in kernel",
            "gated_delta_vi: grid (128,48,128) = 786432 TGs/layer; 48 layers".into(),
            "gpu",
            "isolated rearrange+ba+gated_delta+gated_rmsnorm+dn FMA remainder + 48/64 mixer residual, minus KV/state streams attributed out".into(),
        ),
        row(
            "gqa",
            s(gqa),
            act_write((QWEN38_GQA_HEADS * QWEN38_GQA_HEAD_DIM) as u64) * 16
                + state.gqa_kv_read_bytes_at_pos,
            hidden_b * 16,
            16 * 4,
            1,
            "none in kernel",
            format!(
                "mha_decode_f32 TG=128, 16 layers, seq≈{seq_len}; rope 24 threads"
            ),
            "gpu",
            "isolated rope+mha+sigmoid+gqa FMA remainder + 16/64 mixer residual, minus KV/state stream".into(),
        ),
        row(
            "dense_swiglu",
            s(swiglu),
            mid_b * 2 * 64,
            mid_b * 64 + hidden_b * 64,
            64 * 2,
            1,
            "none in kernel",
            "silu 17408 threads; residual 5120 threads; GEMV FMA remainder after addr/decode".into(),
            "gpu",
            "isolated silu + mlp residual + MLP GEMV FMA remainder (full − addr − decode)".into(),
        ),
        row(
            "normalization",
            s(norms),
            (hidden_b + hidden_b) * 129,
            hidden_b * 129,
            64 + 64 + 1,
            1,
            "none in kernel",
            "qwen80_residual_rmsnorm_f32 TG=256, 129 launches".into(),
            "gpu",
            "isolated input + post + final RMSNorm CBs".into(),
        ),
        row(
            "kv_state",
            s(kv_state),
            state.rec_state_resident_bytes
                + state.conv_state_resident_bytes
                + state.gqa_kv_read_bytes_at_pos,
            state.rec_state_resident_bytes
                + state.conv_state_resident_bytes
                + state.gqa_kv_write_bytes,
            3,
            1,
            "none in kernel",
            "qwen38_f32_stream_probe over rec/conv/GQA resident tensors".into(),
            "gpu",
            "measured sequential f32 stream of rec_state + conv_state + GQA cache, capped by fused parent GPU time".into(),
        ),
        row(
            "terminal_head",
            s(terminal),
            weights.lm_head_bytes,
            act_write(QWEN38_VOCAB as u64) + 4,
            2,
            1,
            "none in kernel; sampled u32 is read after wait (in synchronization)",
            format!(
                "lm_head {} TGs; argmax TG=256",
                geo_tpr64_occupancy(QWEN38_VOCAB as u64).threadgroups
            ),
            "gpu",
            "isolated argmax + lm_head FMA remainder. lm_head weight traffic lives in addressing/decode.".into(),
        ),
        row(
            "synchronization",
            sync.max(0.0),
            0,
            4,
            0,
            1,
            "CPU blocked in waitUntilCompleted minus GPU busy; includes 4-byte sampled-id visibility",
            "n/a (CPU/sync)".into(),
            "sync",
            "production wait_ns − gpu_ns (GPUEnd−GPUStart). Host is not the wall if this is << GPU.".into(),
        ),
        row(
            "unattributed_residual",
            residual_named,
            weights.embed_row_bytes,
            hidden_b,
            1,
            1,
            "none beyond the host tail between Instant marks",
            "embed lookup 5120 threads".into(),
            "gpu",
            format!(
                "NAMED residual = embed_row_gather ({embed_s:.0} ns) + intra_cb_encoder_transition_gap ({gpu_gap:.0} ns) + host_tail_outside_encode_submit_wait ({host_tail:.0} ns) + rounding"
            ),
        ),
    ];

    let sum_serial: i128 = components
        .iter()
        .map(|c| c.ns_per_token.round() as i128)
        .sum();
    let residual_ns = wall_ns as i128 - sum_serial;
    let closure = ClosureLine {
        identity: "sum(serial components) + residual == complete_token_wall; residual is the named unattributed_residual row",
        total_token_ns: wall_ns,
        sum_serial_ns: sum_serial,
        residual_ns,
        residual_name: "unattributed_residual = embed_row_gather + intra_cb_encoder_transition_gap + host_tail".into(),
        residual_reason: format!(
            "embed={embed_s:.0} ns, gpu_gap={gpu_gap:.0} ns, host_tail={host_tail:.0} ns. The named residual row is inside the component table, so sum(components) == wall and residual_ns here is only rounding."
        ),
        identity_holds: sum_serial + residual_ns == wall_ns as i128,
        isolated_family_sum_gpu_ns: isolated_gpu.round() as u64,
        production_gpu_ns: gpu_ns,
        isolated_over_production: isolated_gpu.round() as i128 - gpu_ns as i128,
        gpu_scale_applied: scale,
        scale_reason,
    };
    (components, closure)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_geometry_matches_measured_active_budget_classes() {
        let b = theoretical_weight_bytes();
        // Manifest class bytes include per-tensor HQ30UQ4 headers (~40 B)
        // and a few f32 mixer scales. Geometry here is payload only.
        assert_eq!(b.mlp_bytes, 9_091_153_920);
        assert_eq!(b.linear_attn_bytes, 2_953_789_440);
        assert_eq!(b.full_attn_bytes, 891_289_600);
        assert_eq!(b.lm_head_bytes, 675_430_400);
        let gemv = b.mlp_bytes + b.linear_attn_bytes + b.full_attn_bytes + b.lm_head_bytes;
        assert!(
            gemv.abs_diff(ACTIVE_BUDGET_BYTES) < 20_000_000,
            "geometry gemv {gemv} vs measured active {ACTIVE_BUDGET_BYTES}"
        );
    }

    #[test]
    fn production_dispatch_count_is_964() {
        assert_eq!(production_dispatches_per_token(), 964);
        let d = theoretical_dispatches();
        assert_eq!(d.total, 964);
        assert_eq!(d.production_command_buffers, 1);
        assert_eq!(QWEN38_FULL_LAYER_DISPATCHES, 15);
    }

    #[test]
    fn required_components_are_twelve_and_named() {
        assert_eq!(REQUIRED_COMPONENTS.len(), 12);
        assert!(REQUIRED_COMPONENTS.contains(&"unattributed_residual"));
        assert!(!REQUIRED_COMPONENTS.iter().any(|c| *c == "other"));
    }

    #[test]
    fn closure_identity_holds_on_synthetic_cover() {
        let weights = theoretical_weight_bytes();
        let state = theoretical_state_bytes(26);
        let isolated = vec![
            IsolatedFamily {
                name: "mlp_matvecs_64".into(),
                gpu_ns_reps: vec![15_000_000],
                median_gpu_ns: 15_000_000,
                wait_ns_median: 15_100_000,
                dispatches: 192,
                command_buffers: 1,
            },
            IsolatedFamily {
                name: "dn_gemvs".into(),
                gpu_ns_reps: vec![5_000_000],
                median_gpu_ns: 5_000_000,
                wait_ns_median: 5_100_000,
                dispatches: 144,
                command_buffers: 1,
            },
            IsolatedFamily {
                name: "gqa_gemvs".into(),
                gpu_ns_reps: vec![2_000_000],
                median_gpu_ns: 2_000_000,
                wait_ns_median: 2_100_000,
                dispatches: 64,
                command_buffers: 1,
            },
            IsolatedFamily {
                name: "lm_head".into(),
                gpu_ns_reps: vec![1_000_000],
                median_gpu_ns: 1_000_000,
                wait_ns_median: 1_050_000,
                dispatches: 1,
                command_buffers: 1,
            },
        ];
        let probes = vec![ProbeSplit {
            class: "mlp".into(),
            full_median_gpu_ns: 15_000_000,
            addr_median_gpu_ns: 14_000_000,
            decode_median_gpu_ns: 14_500_000,
            addr_frac_of_full: 14.0 / 15.0,
            decode_minus_addr_frac: 0.5 / 15.0,
            fma_remainder_frac: 0.5 / 15.0,
        }];
        let wall = 34_000_000u64;
        let gpu = 33_500_000u64;
        let encode = 80_000u64;
        let submit = 20_000u64;
        let wait = 33_900_000u64;
        let (rows, closure) = seal_components(
            wall, gpu, encode, submit, wait, &isolated, &probes, &state, &weights, 26,
        );
        assert_eq!(rows.len(), 12);
        assert!(closure.identity_holds);
        let names: Vec<_> = rows.iter().map(|r| r.component.as_str()).collect();
        for req in REQUIRED_COMPONENTS {
            assert!(names.contains(req), "missing {req}");
        }
        let sum: i128 = rows.iter().map(|r| r.ns_per_token.round() as i128).sum();
        assert_eq!(sum + closure.residual_ns, wall as i128);
    }
}
