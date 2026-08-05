//! Broker-kernel A/B promotion gate and cost registry.
//!
//! Planning/groundwork only: ranks and gates kernel *candidates* for the
//! open-source-model brokers (DeepSeek-V4-Flash Terra, Qwen3-Coder Luna,
//! Frankenstein body). This module does **not** touch the DeepSeek-V4 forward
//! lane, does not dispatch Metal, and cannot promote a candidate into serve.
//!
//! Costs are bound to sealed receipts under `receipts/` and the extract at
//! `tools/bench/broker_kernel_ab/receipt_costs.json`.

use serde::{Deserialize, Serialize};

/// Schema for harness receipts.
pub const HARNESS_SCHEMA: &str = "hawking.broker_kernel_ab.v1";

/// Numeric parity classification currently earned on multi-layer DSV4 GPU forward.
pub const MULTI_LAYER_PARITY: &str = "NUMERIC_PARITY_V2_1_ONLY";

/// L0–L1 full forward wall (ms) from `dsv4f_multi_layer_gpu_forward_l0_l1_receipt.json`.
pub const L0_L1_WALL_MS: f64 = 17_984.399_292;
/// Derived ms/layer from that receipt (2 full layers).
pub const L0_L1_MS_PER_LAYER: f64 = L0_L1_WALL_MS / 2.0;
/// L0–L2 wall (ms) from `dsv4f_multi_layer_gpu_forward_bos_l0_l2_receipt.json`.
pub const L0_L2_WALL_MS: f64 = 22_901.087_333;
/// Metal dispatches on L0–L2 BOS multi-layer receipt.
pub const L0_L2_METAL_DISPATCHES: u32 = 276;
/// Command buffers on L0–L2 BOS multi-layer receipt.
pub const L0_L2_COMMAND_BUFFERS: u32 = 26;
/// P6 dispatches per full MoE layer (receipt + execute graph).
pub const P6_DISPATCHES_PER_LAYER: u32 = 60;
/// P7-owned dispatches per full MoE layer.
pub const P7_DISPATCHES_PER_LAYER: u32 = 3;
/// FP4 expert matvec dispatches per full MoE layer (6 experts × W1/W3/W2).
pub const FP4_MATVEC_DISPATCHES_PER_LAYER: u32 = 18;
/// FP8 shared matvec dispatches per full MoE layer.
pub const FP8_SHARED_MATVEC_DISPATCHES_PER_LAYER: u32 = 3;
/// Act-quant dispatches per full MoE layer.
pub const ACT_QUANT_DISPATCHES_PER_LAYER: u32 = 8;
/// Sealed component authority GPU µs for act_quant (simdgroup sweep constant).
pub const ACT_QUANT_AUTHORITY_GPU_US: u64 = 5_967;
/// P4B attention total GPU ms (position-1 complete profile, M3 Ultra).
pub const P4B_ATTENTION_TOTAL_GPU_MS: f64 = 101.55;
/// P4B mHC attn-pre GPU ms (dominant attention kernel).
pub const P4B_MHC_ATTN_PRE_GPU_MS: f64 = 76.608;
/// P4B act_quant aggregate GPU ms (5 dispatches).
pub const P4B_ACT_QUANT_GPU_MS: f64 = 17.463;
/// P4B FP8 control matvec aggregate GPU ms (5 dispatches).
pub const P4B_FP8_CONTROL_GPU_MS: f64 = 1.671;

/// Which broker(s) a kernel family serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerScope {
    /// Shared by Terra + Luna (+ Frankenstein composition).
    Shared,
    /// DeepSeek-V4-Flash Terra (and Frankenstein body reuse).
    TerraDeepSeek,
    /// Qwen3-Coder Luna.
    LunaQwen,
    /// Frankenstein-only composition glue.
    FrankensteinOnly,
}

/// Stable family id used by the harness CLI and plan ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelFamily {
    Fp4ExpertMatvec,
    Fp8ControlMatvec,
    ActQuant,
    MoeGateRoute,
    ExpertGatherCombine,
    MhcControl,
    KvReadWrite,
    LmHeadSample,
    CommandBufferTopology,
}

impl KernelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp4ExpertMatvec => "fp4_expert_matvec",
            Self::Fp8ControlMatvec => "fp8_control_matvec",
            Self::ActQuant => "act_quant",
            Self::MoeGateRoute => "moe_gate_route",
            Self::ExpertGatherCombine => "expert_gather_combine",
            Self::MhcControl => "mhc_control",
            Self::KvReadWrite => "kv_read_write",
            Self::LmHeadSample => "lm_head_sample",
            Self::CommandBufferTopology => "command_buffer_topology",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "fp4_expert_matvec" => Some(Self::Fp4ExpertMatvec),
            "fp8_control_matvec" => Some(Self::Fp8ControlMatvec),
            "act_quant" => Some(Self::ActQuant),
            "moe_gate_route" => Some(Self::MoeGateRoute),
            "expert_gather_combine" => Some(Self::ExpertGatherCombine),
            "mhc_control" => Some(Self::MhcControl),
            "kv_read_write" => Some(Self::KvReadWrite),
            "lm_head_sample" => Some(Self::LmHeadSample),
            "command_buffer_topology" => Some(Self::CommandBufferTopology),
            _ => None,
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::Fp4ExpertMatvec,
            Self::CommandBufferTopology,
            Self::ActQuant,
            Self::Fp8ControlMatvec,
            Self::MhcControl,
            Self::MoeGateRoute,
            Self::ExpertGatherCombine,
            Self::KvReadWrite,
            Self::LmHeadSample,
        ]
    }
}

/// Static ranking entry (impact order for broker serving).
///
/// Serialize-only: entries are compile-time statics with `&'static str` slices
/// that do not implement `Deserialize`.
#[derive(Debug, Clone, Serialize)]
pub struct KernelRankEntry {
    pub rank: u8,
    pub family: KernelFamily,
    pub scope: BrokerScope,
    pub authority_kernel: &'static str,
    pub candidate_kernels: &'static [&'static str],
    pub current_cost: &'static str,
    pub tuning_levers: &'static str,
    pub parity_oracle: &'static str,
}

/// Broker impact ranking (see `KERNEL_BROKERS_TUNING_PLAN.md`).
pub fn broker_kernel_ranking() -> &'static [KernelRankEntry] {
    &[
        KernelRankEntry {
            rank: 1,
            family: KernelFamily::Fp4ExpertMatvec,
            scope: BrokerScope::TerraDeepSeek,
            authority_kernel: "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority",
            candidate_kernels: &["deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_simdgroup_v4_splitk_candidate"],
            current_cost: "18 dispatches/full MoE layer; authority TG=256 serial row reduce; multi-layer ~9 s/layer MoE-dominated (no sealed per-dispatch GPU us yet)",
            tuning_levers: "SIMDgroup width, split-K, packed E2M1/E8M0 loads, TG scale cache, rows-per-TG, expert-wave concurrency",
            parity_oracle: "P5B/P6 component oracle + NumericParity V2.1; exact route IDs",
        },
        KernelRankEntry {
            rank: 2,
            family: KernelFamily::CommandBufferTopology,
            scope: BrokerScope::Shared,
            authority_kernel: "serial_authority_command_buffers",
            candidate_kernels: &["persistent_decode_graph_candidate"],
            current_cost: "L0-L1: 20 CB/waits for 169 disp; L0-L2: 26/276; P4B attn: 33 CB for 33 disp; P6: 4 CB/60 disp",
            tuning_levers: "command-buffer collapse, multi-encoder single wait, pipeline precompile, expert-wave wait collapse (GLM pattern)",
            parity_oracle: "physical trace + zero host intermediate handoff + V2.1 stage outputs",
        },
        KernelRankEntry {
            rank: 3,
            family: KernelFamily::ActQuant,
            scope: BrokerScope::Shared,
            authority_kernel: "deepseek_v4_act_quant_bf16_ue8m0_authority",
            candidate_kernels: &["deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate"],
            current_cost: "P4B: 17.46 ms GPU / 5 disp; sealed authority GPU 5967 us; P6: 8 disp/layer; TG=32",
            tuning_levers: "SIMDgroup block candidate, TG ladder 32..1024, fuse into following matvec when association allows",
            parity_oracle: "byte-exact activation+scale SHA-256 (act_quant sweep contract)",
        },
        KernelRankEntry {
            rank: 4,
            family: KernelFamily::Fp8ControlMatvec,
            scope: BrokerScope::TerraDeepSeek,
            authority_kernel: "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority",
            candidate_kernels: &["deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate"],
            current_cost: "P4B: 1.67 ms GPU / 5 disp, 75.5 MB read; P6 shared: 3 disp/layer; TG=256",
            tuning_levers: "split-K, SIMDgroup v4, packed FP8 loads, dual-issue with act_quant",
            parity_oracle: "NumericParity V2.1 op-local vs FP64; QAT input storage exact",
        },
        KernelRankEntry {
            rank: 5,
            family: KernelFamily::MhcControl,
            scope: BrokerScope::TerraDeepSeek,
            authority_kernel: "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority",
            candidate_kernels: &[
                "deepseek_v4_p4b_hc_post_comb_darwin_dd_candidate",
                "deepseek_v4_p7_mhc_ffn_pre_authority",
            ],
            current_cost: "P4B mHC attn-pre: 76.61 ms GPU / 2 disp (75% of attention GPU); P7: 3 disp/layer, pre 1-thread",
            tuning_levers: "parallel Sinkhorn/mix under Darwin DD exp only; never fast-exp for control domain",
            parity_oracle: "P4B/P7 control-domain + NumericParity V2.1 residuals",
        },
        KernelRankEntry {
            rank: 6,
            family: KernelFamily::MoeGateRoute,
            scope: BrokerScope::Shared,
            authority_kernel: "deepseek_v4_p6a_gate_bf16_matvec_authority",
            candidate_kernels: &[
                "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate",
                "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority",
                "moe_topk_gate",
            ],
            current_cost: "2 disp/layer (gate+route); gate simdgroup=32 exact; learned-bias sealed 1 disp (compose blocked for L3+ P6)",
            tuning_levers: "gate reduction association candidates; learned-bias two-phase compose; Qwen top-8/128 moe_topk_gate",
            parity_oracle: "exact expert IDs + V2.1 scores/weights",
        },
        KernelRankEntry {
            rank: 7,
            family: KernelFamily::ExpertGatherCombine,
            scope: BrokerScope::Shared,
            authority_kernel: "deepseek_v4_p6a_route6_shared_combine_bf16_authority",
            candidate_kernels: &["moe_gather_combine", "expert_wave_wait_collapse_candidate"],
            current_cost: "1 combine disp/layer; projections already concurrent; waits not collapsed to 1 drain/layer",
            tuning_levers: "weighted gather fusion into W2 epilogue; expert-wave wait collapse",
            parity_oracle: "exact combine order + single route-weight application",
        },
        KernelRankEntry {
            rank: 8,
            family: KernelFamily::KvReadWrite,
            scope: BrokerScope::Shared,
            authority_kernel: "deepseek_v4_p4b_kv_cache_write_bf16_authority",
            candidate_kernels: &["kv_scatter_append_multiseq", "rope_qk_kv_append_fused"],
            current_cost: "pos1 write 0.01 ms GPU; sparse sink 0.10 ms; long-ctx cost unmeasured (Luna 256K pending)",
            tuning_levers: "fused rope+KV append, paged/windowed sparse KV, INT4/F16 KV variants with sealed parity",
            parity_oracle: "exact KV row storage where claimed; V2.1 attention outputs",
        },
        KernelRankEntry {
            rank: 9,
            family: KernelFamily::LmHeadSample,
            scope: BrokerScope::Shared,
            authority_kernel: "sample_argmax_f32",
            candidate_kernels: &["gemv_f16_argmax_metal_pinned", "lm_head_simdmat_candidate"],
            current_cost: "no sealed DSV4 lm_head GPU us (greedy_token_produced=false on multi-layer receipts)",
            tuning_levers: "simdmat lm_head, device argmax, vocab prune under policy",
            parity_oracle: "NumericParity V2.1 logits + exact greedy argmax",
        },
    ]
}

/// Inputs to the promotion gate. Speed fields are optional so parity-only
/// dry runs remain valid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTrialInput {
    pub family: KernelFamily,
    pub authority_kernel: String,
    pub candidate_kernel: String,
    /// True only when the sealed / V2.1 oracle accepts the candidate.
    pub parity_pass: bool,
    /// Optional short human reason when parity fails.
    pub parity_detail: Option<String>,
    /// Authority GPU p50 µs (when timed).
    pub authority_gpu_p50_us: Option<u64>,
    /// Candidate GPU p50 µs (when timed).
    pub candidate_gpu_p50_us: Option<u64>,
}

/// Verdicts the scaffold may emit. `ServePromote` is intentionally absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionVerdict {
    /// Candidate failed the sealed / V2.1 oracle.
    RejectParity,
    /// Parity ok but no measured speed win (or speed not measured).
    RejectNoWin,
    /// Parity ok and candidate is faster — still **not** served; manual only.
    CandidateReady,
}

/// Full decision record for a receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionDecision {
    pub schema: &'static str,
    pub multi_layer_parity_class: &'static str,
    pub family: KernelFamily,
    pub authority_kernel: String,
    pub candidate_kernel: String,
    pub parity_pass: bool,
    pub parity_detail: Option<String>,
    pub speed_ratio: Option<f64>,
    pub speed_improved: bool,
    pub verdict: PromotionVerdict,
    pub serve_promoted: bool,
    pub note: &'static str,
}

const CANDIDATE_READY_NOTE: &str =
    "Parity passed and candidate is faster than authority. Manual integration only — scaffold never flips serve defaults.";
const REJECT_PARITY_NOTE: &str =
    "Candidate rejected: sealed/V2.1 parity failed. Speed is irrelevant.";
const REJECT_NO_WIN_NOTE: &str =
    "Parity ok but no measured speed improvement; record only, do not promote.";

/// Decide whether a candidate is eligible for *manual* consideration.
///
/// # Invariants
/// - `serve_promoted` is **always** false.
/// - Any `parity_pass == false` → `RejectParity`.
/// - Speed win without parity cannot occur (parity checked first).
pub fn decide_promotion(input: AbTrialInput) -> PromotionDecision {
    let speed_ratio = match (input.authority_gpu_p50_us, input.candidate_gpu_p50_us) {
        (Some(a), Some(c)) if a > 0 => Some(c as f64 / a as f64),
        _ => None,
    };
    // Strict improvement: candidate p50 must be strictly below authority.
    let speed_improved = matches!(
        (input.authority_gpu_p50_us, input.candidate_gpu_p50_us),
        (Some(a), Some(c)) if c < a
    );

    let (verdict, note) = if !input.parity_pass {
        (PromotionVerdict::RejectParity, REJECT_PARITY_NOTE)
    } else if !speed_improved {
        (PromotionVerdict::RejectNoWin, REJECT_NO_WIN_NOTE)
    } else {
        (PromotionVerdict::CandidateReady, CANDIDATE_READY_NOTE)
    };

    PromotionDecision {
        schema: HARNESS_SCHEMA,
        multi_layer_parity_class: MULTI_LAYER_PARITY,
        family: input.family,
        authority_kernel: input.authority_kernel,
        candidate_kernel: input.candidate_kernel,
        parity_pass: input.parity_pass,
        parity_detail: input.parity_detail,
        speed_ratio,
        speed_improved,
        verdict,
        serve_promoted: false,
        note,
    }
}

/// Summary of receipt-backed layer costs for harness banners / receipts.
#[derive(Debug, Clone, Serialize)]
pub struct LayerCostSnapshot {
    pub l0_l1_wall_ms: f64,
    pub l0_l1_ms_per_layer: f64,
    pub l0_l2_wall_ms: f64,
    pub l0_l2_metal_dispatches: u32,
    pub l0_l2_command_buffers: u32,
    pub p6_dispatches_per_layer: u32,
    pub fp4_matvec_dispatches_per_layer: u32,
    pub act_quant_authority_gpu_us: u64,
    pub p4b_attention_total_gpu_ms: f64,
    pub p4b_mhc_attn_pre_gpu_ms: f64,
    pub parity_class: &'static str,
}

pub fn layer_cost_snapshot() -> LayerCostSnapshot {
    LayerCostSnapshot {
        l0_l1_wall_ms: L0_L1_WALL_MS,
        l0_l1_ms_per_layer: L0_L1_MS_PER_LAYER,
        l0_l2_wall_ms: L0_L2_WALL_MS,
        l0_l2_metal_dispatches: L0_L2_METAL_DISPATCHES,
        l0_l2_command_buffers: L0_L2_COMMAND_BUFFERS,
        p6_dispatches_per_layer: P6_DISPATCHES_PER_LAYER,
        fp4_matvec_dispatches_per_layer: FP4_MATVEC_DISPATCHES_PER_LAYER,
        act_quant_authority_gpu_us: ACT_QUANT_AUTHORITY_GPU_US,
        p4b_attention_total_gpu_ms: P4B_ATTENTION_TOTAL_GPU_MS,
        p4b_mhc_attn_pre_gpu_ms: P4B_MHC_ATTN_PRE_GPU_MS,
        parity_class: MULTI_LAYER_PARITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_serve_promotes() {
        let d = decide_promotion(AbTrialInput {
            family: KernelFamily::ActQuant,
            authority_kernel: "auth".into(),
            candidate_kernel: "cand".into(),
            parity_pass: true,
            parity_detail: None,
            authority_gpu_p50_us: Some(1000),
            candidate_gpu_p50_us: Some(100),
        });
        assert!(!d.serve_promoted);
        assert_eq!(d.verdict, PromotionVerdict::CandidateReady);
    }

    #[test]
    fn parity_failure_ignores_speed() {
        let d = decide_promotion(AbTrialInput {
            family: KernelFamily::Fp4ExpertMatvec,
            authority_kernel: "auth".into(),
            candidate_kernel: "cand".into(),
            parity_pass: false,
            parity_detail: Some("rel_l2".into()),
            authority_gpu_p50_us: Some(10_000),
            candidate_gpu_p50_us: Some(1),
        });
        assert_eq!(d.verdict, PromotionVerdict::RejectParity);
        // Speed would look like a win, but parity vetoes promotion.
        assert!(d.speed_improved);
        assert!(!d.serve_promoted);
    }
}
