//! Per-served-weight and per-moved-bit units for a TOKEN_NS document.
//!
//! tok/s is model-size-dependent. The comparable units are femtoseconds per
//! served weight and picoseconds per moved bit. Both are **amortized
//! throughput-derived service times**, not latencies. A single weight's DRAM
//! round trip is still ~100 ns; femtoseconds appear only because thousands of
//! weights are in flight at once.

use serde::{Deserialize, Serialize};

use super::energy::{EnergyReport, ENERGY_FILL_COMMAND, SINGLE_WEIGHT_DRAM_ROUND_TRIP_NS};

/// Published peak of this machine's M3 Ultra 96 GB unified memory.
pub const M3_ULTRA_96GB_PEAK_BYTES_PER_S: f64 = 819.0e9;

/// 1 / (819e9 B/s × 8 bit/B) in picoseconds. Brief rounds this to 0.153.
pub const HARDWARE_PS_PER_BIT: f64 = 1.0e12 / (M3_ULTRA_96GB_PEAK_BYTES_PER_S * 8.0);

/// JSON / field label. A reader must not be able to mistake this for latency.
pub const FS_PER_WEIGHT_SERVED_FIELD: &str =
    "fs_per_weight_served (amortized throughput-derived, NOT latency)";

pub const AMORTIZED_CAVEAT: &str = "fs_per_weight_served is amortized service time under concurrency, NOT latency. A single weight's real DRAM round trip is ~100 ns. Femtoseconds appear only because thousands of weights are in flight at once. Do not read this number as a claim of femtosecond latency.";

/// Q80 mixed-representation BPW published in PHYSICAL_FLOOR.json.
pub const Q80_MIXED_BPW: f64 = 1.392467;
/// Q80 uniform-Q4 vehicle physical BPW.
pub const Q80_UNIFORM_Q4_BPW: f64 = 4.259241;

/// Attention products used by [`crate::model::qwen80_token_ns_ledger::theoretical_weight_bytes_per_token`].
const Q80_DN_QKVZ_ROWS: u64 = 12_288;
const Q80_DN_BA_ROWS: u64 = 64;
const Q80_DN_OUT_COLS: u64 = 4_096;
const Q80_GQA_Q_ROWS: u64 = 8_192;
const Q80_GQA_KV_ROWS: u64 = 512;
const Q80_GQA_O_COLS: u64 = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelId {
    Q80,
    Dsv4f,
    Qwen38,
    Unknown,
}

impl ModelId {
    pub fn parse(model: &str) -> Self {
        match model {
            "q80" | "qwen80" | "Q80" => Self::Q80,
            "dsv4f" | "dsv" | "deepseek_v4" | "DSV4F" => Self::Dsv4f,
            "qwen38" | "qwen3.8" | "Qwen38" | "QWEN38" => Self::Qwen38,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightComponent {
    pub name: String,
    pub weights: u64,
    pub how: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActiveWeightGeometry {
    pub model: String,
    pub active_weights_per_token: u64,
    pub components: Vec<WeightComponent>,
    pub excluded_from_denominator: Vec<WeightComponent>,
    pub derivation: String,
    pub matches_physical_floor_q80: Option<bool>,
}

impl ActiveWeightGeometry {
    pub fn for_model(id: ModelId) -> Self {
        match id {
            ModelId::Q80 => q80_geometry(),
            ModelId::Dsv4f => dsv4f_geometry(),
            ModelId::Qwen38 => qwen38_geometry(),
            ModelId::Unknown => ActiveWeightGeometry {
                model: "unknown".into(),
                active_weights_per_token: 0,
                components: Vec::new(),
                excluded_from_denominator: Vec::new(),
                derivation: "unknown model; active_weights_per_token cannot be derived".into(),
                matches_physical_floor_q80: None,
            },
        }
    }
}

/// Q80 served GEMM weights for one decode token.
///
/// Denominator matches PHYSICAL_FLOOR.json `active_params_per_token` =
/// 3,562,274,816: attention + routed(top-10) + shared + router + lm_head.
/// Embed (one row) and the shared-expert gate are served but omitted from
/// that published number (100,352 weights, 0.0028%).
pub fn q80_geometry() -> ActiveWeightGeometry {
    use crate::model::qwen80_48_layer_execution_schedule::{
        QWEN80_DELTANET_LAYERS, QWEN80_EXPERTS, QWEN80_GQA_LAYERS, QWEN80_HIDDEN, QWEN80_LAYERS,
        QWEN80_TOP_K,
    };
    use crate::model::qwen80_source_bf16_layer_major::{
        QWEN80_MOE_INTERMEDIATE, QWEN80_SHARED_EXPERT_INTERMEDIATE, QWEN80_VOCAB,
    };

    let hidden = QWEN80_HIDDEN as u64;
    let layers = QWEN80_LAYERS as u64;
    let dn_layers = QWEN80_DELTANET_LAYERS as u64;
    let gqa_layers = QWEN80_GQA_LAYERS as u64;
    let experts = QWEN80_EXPERTS as u64;
    let top_k = QWEN80_TOP_K as u64;
    let moe_i = QWEN80_MOE_INTERMEDIATE as u64;
    let shared_i = QWEN80_SHARED_EXPERT_INTERMEDIATE as u64;
    let vocab = QWEN80_VOCAB as u64;

    let dn = Q80_DN_QKVZ_ROWS * hidden + Q80_DN_BA_ROWS * hidden + hidden * Q80_DN_OUT_COLS;
    let gqa = Q80_GQA_Q_ROWS * hidden
        + Q80_GQA_KV_ROWS * hidden
        + Q80_GQA_KV_ROWS * hidden
        + hidden * Q80_GQA_O_COLS;
    let attention = dn_layers * dn + gqa_layers * gqa;
    let routed = layers * top_k * 3 * moe_i * hidden;
    let shared = layers * 3 * shared_i * hidden;
    let router = layers * experts * hidden;
    let lm_head = vocab * hidden;
    let embed_row = hidden;
    let shared_gate = layers * hidden;

    let active = attention + routed + shared + router + lm_head;
    ActiveWeightGeometry {
        model: "q80".into(),
        active_weights_per_token: active,
        components: vec![
            WeightComponent {
                name: "attention_deltanet_gqa".into(),
                weights: attention,
                how: format!(
                    "{dn_layers} DeltaNet (qkvz {Q80_DN_QKVZ_ROWS}×{hidden} + ba {Q80_DN_BA_ROWS}×{hidden} + out {hidden}×{Q80_DN_OUT_COLS}) + {gqa_layers} gated-GQA (q {Q80_GQA_Q_ROWS}×{hidden} + k/v {Q80_GQA_KV_ROWS}×{hidden} + o {hidden}×{Q80_GQA_O_COLS})"
                ),
            },
            WeightComponent {
                name: "routed_expert".into(),
                weights: routed,
                how: format!("{layers} layers × top-{top_k} × 3 proj × {moe_i} × {hidden}"),
            },
            WeightComponent {
                name: "shared_expert".into(),
                weights: shared,
                how: format!("{layers} layers × 1 shared × 3 proj × {shared_i} × {hidden}"),
            },
            WeightComponent {
                name: "router".into(),
                weights: router,
                how: format!("{layers} layers × {experts} experts × {hidden}"),
            },
            WeightComponent {
                name: "lm_head".into(),
                weights: lm_head,
                how: format!("full GEMV {vocab} × {hidden}"),
            },
        ],
        excluded_from_denominator: vec![
            WeightComponent {
                name: "embed_row".into(),
                weights: embed_row,
                how: "one vocab row is looked up; PHYSICAL_FLOOR omitted it".into(),
            },
            WeightComponent {
                name: "shared_expert_gate".into(),
                weights: shared_gate,
                how: format!("{layers} × {hidden}; PHYSICAL_FLOOR omitted it"),
            },
        ],
        derivation: "attention + routed(top_k) + shared + router + lm_head. Matches receipts/ascent-2026-08-16/PHYSICAL_FLOOR.json active_params_per_token.".into(),
        matches_physical_floor_q80: Some(active == 3_562_274_816),
    }
}

/// DSV4F served GEMM weights for one decode token.
///
/// Same class of denominator as Q80: MLA + routed(top-6 of 256) + shared
/// expert + router + lm_head. Full embed is a lookup of one row, not 529 M
/// served weights. Indexer/compressor/mHC/norms are listed as exclusions so
/// they stay visible without changing the comparable unit.
pub fn dsv4f_geometry() -> ActiveWeightGeometry {
    use crate::gravity_deepseek_v4_layer_source_anchors::{
        DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT, DSV4F_LAYER_SOURCE_ANCHOR_HIDDEN_SIZE,
        DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT, DSV4F_LAYER_SOURCE_ANCHOR_TOP_K,
    };

    let layers = DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT as u64;
    let hidden = DSV4F_LAYER_SOURCE_ANCHOR_HIDDEN_SIZE as u64;
    let routed_n = DSV4F_LAYER_SOURCE_ANCHOR_ROUTED_EXPERT_COUNT as u64;
    let top_k = DSV4F_LAYER_SOURCE_ANCHOR_TOP_K as u64;
    let shared_n = 1u64;
    // Logical SwiGLU width from the sealed pair shapes (FP8 shared w1 [2048,4096],
    // FP4 routed w1 packed as [2048,2048] e2m1fn_x2 → logical 2048×4096).
    let moe_i = 2_048u64;
    let vocab = 129_280u64;
    // MLA pair shapes in gravity_deepseek_v4_layer_source_anchors.rs.
    let wq_a = 1_024u64 * 4_096;
    let wq_b = 32_768u64 * 1_024;
    let wkv = 512u64 * 4_096;
    let wo_a = 8_192u64 * 4_096;
    let wo_b = 4_096u64 * 8_192;
    let mla_layer = wq_a + wq_b + wkv + wo_a + wo_b;
    let mla = layers * mla_layer;
    let routed = layers * top_k * 3 * moe_i * hidden;
    let shared = layers * shared_n * 3 * moe_i * hidden;
    let router = layers * routed_n * hidden;
    let lm_head = vocab * hidden;
    let embed_row = hidden;
    let embed_table = vocab * hidden;

    let active = mla + routed + shared + router + lm_head;
    ActiveWeightGeometry {
        model: "dsv4f".into(),
        active_weights_per_token: active,
        components: vec![
            WeightComponent {
                name: "mla".into(),
                weights: mla,
                how: format!(
                    "{layers} layers × (wq_a 1024×4096 + wq_b 32768×1024 + wkv 512×4096 + wo_a 8192×4096 + wo_b 4096×8192) = {mla_layer}/layer"
                ),
            },
            WeightComponent {
                name: "routed_expert".into(),
                weights: routed,
                how: format!("{layers} layers × top-{top_k} of {routed_n} × 3 proj × {moe_i} × {hidden}"),
            },
            WeightComponent {
                name: "shared_expert".into(),
                weights: shared,
                how: format!("{layers} layers × {shared_n} shared × 3 proj × {moe_i} × {hidden}"),
            },
            WeightComponent {
                name: "router".into(),
                weights: router,
                how: format!("{layers} layers × {routed_n} × {hidden}"),
            },
            WeightComponent {
                name: "lm_head".into(),
                weights: lm_head,
                how: format!("full GEMV {vocab} × {hidden}"),
            },
        ],
        excluded_from_denominator: vec![
            WeightComponent {
                name: "embed_row".into(),
                weights: embed_row,
                how: "one vocab row is looked up per token".into(),
            },
            WeightComponent {
                name: "embed_full_table_not_served".into(),
                weights: embed_table,
                how: "dsv-resident-gravity lower bound counted the full table; this denominator does not, because a lookup does not serve vocab×hidden weights".into(),
            },
            WeightComponent {
                name: "indexer_compressor_mhc_norms".into(),
                weights: 0,
                how: "ratio-4/128 indexer+compressor, mHC hc_fn, and RMSNorms fire on some/all layers but are kept out of the Q80-comparable denominator. They are not zero cost; they are a different organ class.".into(),
            },
        ],
        derivation: "Same class as Q80 PHYSICAL_FLOOR: attention (MLA) + routed(top_k) + shared + router + lm_head. 43 layers, top-6 of 256, one shared expert.".into(),
        matches_physical_floor_q80: None,
    }
}

/// Qwen3.8 served GEMM weights for one decode token.
///
/// Dense model: every GEMV is read every token except the embedding table
/// (one row). No experts.
pub fn qwen38_geometry() -> ActiveWeightGeometry {
    use crate::model::qwen38_geometry::{
        QWEN38_DELTANET_LAYERS, QWEN38_GQA_LAYERS, QWEN38_HIDDEN, QWEN38_INTERMEDIATE,
        QWEN38_LAYERS, QWEN38_VOCAB,
    };
    let h = QWEN38_HIDDEN as u64;
    let mid = QWEN38_INTERMEDIATE as u64;
    let layers = QWEN38_LAYERS as u64;
    let dn = QWEN38_DELTANET_LAYERS as u64;
    let gqa = QWEN38_GQA_LAYERS as u64;
    let vocab = QWEN38_VOCAB as u64;
    let mlp = layers * (mid * h * 2 + h * mid);
    let linear = dn * (16_384 * h + 96 * h + h * 6_144);
    let full = gqa * (12_288 * h + 1_024 * h * 2 + h * 6_144);
    let lm_head = vocab * h;
    let embed_row = h;
    let embed_table = vocab * h;
    let active = mlp + linear + full + lm_head;
    ActiveWeightGeometry {
        model: "qwen38".into(),
        active_weights_per_token: active,
        components: vec![
            WeightComponent {
                name: "dense_swiglu".into(),
                weights: mlp,
                how: format!("{layers} layers × (gate+up {mid}×{h} + down {h}×{mid})"),
            },
            WeightComponent {
                name: "deltanet_gemv".into(),
                weights: linear,
                how: format!("{dn} DeltaNet layers × (qkvz 16384×{h} + ba 96×{h} + out {h}×6144)"),
            },
            WeightComponent {
                name: "gqa_gemv".into(),
                weights: full,
                how: format!("{gqa} GQA layers × (q 12288×{h} + k/v 1024×{h} + o {h}×6144)"),
            },
            WeightComponent {
                name: "lm_head".into(),
                weights: lm_head,
                how: format!("full GEMV {vocab} × {h}"),
            },
        ],
        excluded_from_denominator: vec![
            WeightComponent {
                name: "embed_row".into(),
                weights: embed_row,
                how: "one vocab row is gathered per token".into(),
            },
            WeightComponent {
                name: "embed_full_table_not_served".into(),
                weights: embed_table,
                how: "dense model reads every weight except the embedding table".into(),
            },
        ],
        derivation: "Dense Qwen3.8: MLP + linear-attn GEMVs + GQA GEMVs + lm_head. Embed table excluded.".into(),
        matches_physical_floor_q80: None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PublishedFloor {
    pub bpw: f64,
    pub bytes_per_token: u64,
    pub fs_per_weight_floor: f64,
    pub token_floor_us: f64,
    pub ceiling_tok_s: f64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServedWeightHonesty {
    pub not_latency: bool,
    pub field_label: String,
    pub caveat: String,
    pub single_weight_dram_round_trip_ns_approx: f64,
}

impl ServedWeightHonesty {
    pub fn standard() -> Self {
        Self {
            not_latency: true,
            field_label: FS_PER_WEIGHT_SERVED_FIELD.to_owned(),
            caveat: AMORTIZED_CAVEAT.to_owned(),
            single_weight_dram_round_trip_ns_approx: SINGLE_WEIGHT_DRAM_ROUND_TRIP_NS,
        }
    }
}

/// Physical units attached to every TOKEN_NS document.
#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServedWeightMetrics {
    pub active_weights_per_token: u64,
    pub bits_moved_per_token: u64,
    pub bytes_moved_per_token: u64,
    pub peak_bandwidth_bytes_per_s: f64,
    pub hardware_ps_per_bit: f64,
    pub token_ns: u64,
    pub bpw_this_run: Option<f64>,
    #[serde(rename = "fs_per_weight_served (amortized throughput-derived, NOT latency)")]
    pub fs_per_weight_served: f64,
    pub ps_per_bit_moved: Option<f64>,
    pub fs_per_weight_floor: Option<f64>,
    pub distance_from_floor: Option<f64>,
    /// Picojoules per served weight. Spelled `pJ` because that is the unit.
    #[allow(non_snake_case)]
    pub pJ_per_weight_served: Option<f64>,
    pub energy_available: bool,
    pub energy: EnergyReport,
    pub honesty: ServedWeightHonesty,
    pub geometry: ActiveWeightGeometry,
    pub reference_floors: Vec<PublishedFloor>,
}

impl Default for ServedWeightMetrics {
    fn default() -> Self {
        Self::unavailable("not yet sealed")
    }
}

impl ServedWeightMetrics {
    pub fn unavailable(why: &str) -> Self {
        Self {
            active_weights_per_token: 0,
            bits_moved_per_token: 0,
            bytes_moved_per_token: 0,
            peak_bandwidth_bytes_per_s: M3_ULTRA_96GB_PEAK_BYTES_PER_S,
            hardware_ps_per_bit: HARDWARE_PS_PER_BIT,
            token_ns: 0,
            bpw_this_run: None,
            fs_per_weight_served: 0.0,
            ps_per_bit_moved: None,
            fs_per_weight_floor: None,
            distance_from_floor: None,
            pJ_per_weight_served: None,
            energy_available: false,
            energy: EnergyReport::unavailable(why),
            honesty: ServedWeightHonesty::standard(),
            geometry: ActiveWeightGeometry::for_model(ModelId::Unknown),
            reference_floors: Vec::new(),
        }
    }

    pub fn compute(
        model: &str,
        token_ns: u64,
        bytes_moved_per_token: u64,
        joules_per_token: Option<f64>,
        energy_reason_if_absent: Option<&str>,
    ) -> Self {
        let geometry = ActiveWeightGeometry::for_model(ModelId::parse(model));
        let w = geometry.active_weights_per_token;
        let bits = bytes_moved_per_token.saturating_mul(8);
        let fs_per_weight_served = if w == 0 {
            0.0
        } else {
            // token_ns * 1e-9 s / w * 1e15 fs = token_ns * 1e6 / w
            token_ns as f64 * 1.0e6 / w as f64
        };
        let (ps_per_bit_moved, fs_per_weight_floor, distance_from_floor, bpw_this_run) =
            if bytes_moved_per_token == 0 || w == 0 {
                (None, None, None, None)
            } else {
                let ps_bit = token_ns as f64 * 1.0e3 / bits as f64;
                let floor = bytes_moved_per_token as f64
                    / M3_ULTRA_96GB_PEAK_BYTES_PER_S
                    / w as f64
                    * 1.0e15;
                let dist = if floor > 0.0 {
                    Some(fs_per_weight_served / floor)
                } else {
                    None
                };
                let bpw = bits as f64 / w as f64;
                (Some(ps_bit), Some(floor), dist, Some(bpw))
            };
        let energy = match joules_per_token {
            Some(j) => EnergyReport::from_caller_joules(j),
            None => EnergyReport::unavailable(energy_reason_if_absent.unwrap_or(
                "no joules_per_token supplied; historical TOKEN_NS ledgers have no energy wrap",
            )),
        };
        let pj = joules_per_token.and_then(|j| {
            if w == 0 {
                None
            } else {
                Some(j * 1.0e12 / w as f64)
            }
        });
        Self {
            active_weights_per_token: w,
            bits_moved_per_token: bits,
            bytes_moved_per_token,
            peak_bandwidth_bytes_per_s: M3_ULTRA_96GB_PEAK_BYTES_PER_S,
            hardware_ps_per_bit: HARDWARE_PS_PER_BIT,
            token_ns,
            bpw_this_run,
            fs_per_weight_served,
            ps_per_bit_moved,
            fs_per_weight_floor,
            distance_from_floor,
            pJ_per_weight_served: pj,
            energy_available: energy.energy_available,
            energy,
            honesty: ServedWeightHonesty::standard(),
            reference_floors: reference_floors_for(&geometry),
            geometry,
        }
    }
}

fn reference_floors_for(geometry: &ActiveWeightGeometry) -> Vec<PublishedFloor> {
    if geometry.model != "q80" || geometry.active_weights_per_token == 0 {
        return Vec::new();
    }
    let w = geometry.active_weights_per_token as f64;
    let mk = |bpw: f64, source: &'static str| {
        let bytes = (w * bpw / 8.0).round() as u64;
        let token_s = bytes as f64 / M3_ULTRA_96GB_PEAK_BYTES_PER_S;
        PublishedFloor {
            bpw,
            bytes_per_token: bytes,
            fs_per_weight_floor: token_s / w * 1.0e15,
            token_floor_us: token_s * 1.0e6,
            ceiling_tok_s: if token_s > 0.0 { 1.0 / token_s } else { 0.0 },
            source: source.to_owned(),
        }
    };
    vec![
        mk(
            Q80_UNIFORM_Q4_BPW,
            "PHYSICAL_FLOOR.json q80_q4_vehicle_4_259241",
        ),
        mk(Q80_MIXED_BPW, "PHYSICAL_FLOOR.json q80_mixed_1_392467"),
        mk(1.0, "PHYSICAL_FLOOR.json 1.000000 BPW row"),
        mk(0.5, "PHYSICAL_FLOOR.json 0.500000 BPW row"),
        mk(0.25, "PHYSICAL_FLOOR.json 0.250000 BPW row"),
    ]
}

/// Bytes a human should feed powermetrics / IOReport into later, without
/// re-deriving geometry.
pub fn energy_fill_command() -> &'static str {
    ENERGY_FILL_COMMAND
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q80_active_weights_match_physical_floor() {
        let g = q80_geometry();
        assert_eq!(g.active_weights_per_token, 3_562_274_816);
        assert_eq!(g.matches_physical_floor_q80, Some(true));
        let extra: u64 = g.excluded_from_denominator.iter().map(|c| c.weights).sum();
        assert_eq!(extra, 100_352);
    }

    #[test]
    fn dsv4f_active_weights_are_geometry_not_a_constant_copy() {
        let g = dsv4f_geometry();
        assert_eq!(g.active_weights_per_token, 12_748_587_008);
        let mla = g
            .components
            .iter()
            .find(|c| c.name == "mla")
            .unwrap()
            .weights;
        let routed = g
            .components
            .iter()
            .find(|c| c.name == "routed_expert")
            .unwrap()
            .weights;
        let shared = g
            .components
            .iter()
            .find(|c| c.name == "shared_expert")
            .unwrap()
            .weights;
        assert_eq!(mla, 43 * 106_954_752);
        assert_eq!(routed, 43 * 6 * 3 * 2048 * 4096);
        assert_eq!(shared, 43 * 3 * 2048 * 4096);
        assert!(g
            .excluded_from_denominator
            .iter()
            .any(|c| c.name == "embed_full_table_not_served"));
    }

    #[test]
    fn mixed_bpw_floor_is_212_fs() {
        let w = 3_562_274_816u64;
        let bytes = 620_043_766u64;
        let floor = bytes as f64 / M3_ULTRA_96GB_PEAK_BYTES_PER_S / w as f64 * 1.0e15;
        assert!(
            (floor - 212.53).abs() < 0.02,
            "fs/weight floor at 1.392467 BPW = {floor}"
        );
        let q4_bytes = 1_896_573_369u64;
        let q4_floor = q4_bytes as f64 / M3_ULTRA_96GB_PEAK_BYTES_PER_S / w as f64 * 1.0e15;
        assert!(
            (q4_floor - 650.07).abs() < 0.02,
            "fs/weight floor at 4.259241 BPW = {q4_floor}"
        );
    }

    #[test]
    fn hardware_ps_per_bit_is_the_published_0_153() {
        assert!((HARDWARE_PS_PER_BIT - 0.152625).abs() < 1e-6);
        assert!((HARDWARE_PS_PER_BIT - 0.153).abs() < 0.001);
    }

    #[test]
    fn q80_403ms_is_113_ps_amortized_not_latency() {
        let m = ServedWeightMetrics::compute("q80", 403_000_000, 620_043_766, None, None);
        // 403e6 ns * 1e6 / 3.562274816e9 = 113129.3 fs = 113.13 ps
        assert!((m.fs_per_weight_served - 113_129.3).abs() < 1.0);
        assert!(m.honesty.not_latency);
        assert!(m.honesty.caveat.contains("NOT latency"));
        assert!(m.honesty.field_label.contains("NOT latency"));
        assert!(!m.energy_available);
        assert!(m.pJ_per_weight_served.is_none());
    }

    #[test]
    fn pj_only_when_joules_supplied() {
        let m = ServedWeightMetrics::compute("q80", 559_171_655, 1_892_511_808, Some(2.0), None);
        assert!(m.energy_available);
        let expected = 2.0 * 1.0e12 / 3_562_274_816.0;
        assert!((m.pJ_per_weight_served.unwrap() - expected).abs() < 1e-6);
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains(FS_PER_WEIGHT_SERVED_FIELD));
        assert!(json.contains("NOT latency"));
    }

    #[test]
    fn missing_bytes_still_emits_fs_per_weight() {
        let m = ServedWeightMetrics::compute("dsv4f", 1_037_764_208, 0, None, None);
        assert_eq!(m.active_weights_per_token, 12_748_587_008);
        assert!(m.fs_per_weight_served > 0.0);
        assert!(m.ps_per_bit_moved.is_none());
        assert!(m.fs_per_weight_floor.is_none());
    }
}
