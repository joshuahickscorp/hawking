//! Honest roof for Qwen3.8 `weight_addressing`.
//!
//! The campaign treated 411.51 GB/s (the 512 MiB point of a sequential
//! `unique_once` read-reduce) as the decode ceiling, then reported
//! `weight_addressing` as 97.6 % of that ceiling by dividing the 13.6 GB
//! active budget by **total** production GPU time. This module:
//!
//! 1. Adjudicates the three disagreeing byte counts.
//! 2. Attributes GEMV bytes to the time that actually moves them.
//! 3. On macOS, measures bandwidth vs working-set size on the **real**
//!    access pattern: `qwen_uniform_q4_group64_matvec_geo_tpr64_tg128`
//!    (and its addr/decode probes), with GPU timestamps from a completed
//!    `MTLCommandBuffer`.
//!
//! Does not change codecs, kernels, or the decode path.

use crate::model::qwen38_geometry::{
    QWEN38_BA_ROWS, QWEN38_DELTANET_LAYERS, QWEN38_GQA_LAYERS, QWEN38_HIDDEN, QWEN38_INTERMEDIATE,
    QWEN38_KV_PROJ_ROWS, QWEN38_LAYERS, QWEN38_O_PROJ_COLS, QWEN38_Q_PROJ_ROWS, QWEN38_QKVZ_ROWS,
    QWEN38_VOCAB,
};
use crate::model::qwen38_token_ns_ledger::{
    theoretical_weight_bytes, ACTIVE_BUDGET_BYTES, HONEST_DECODE_CEILING_GB_S,
};
use crate::model::qwen_complete_binary::{
    UNIFORM_Q4_CODE_BYTES_PER_GROUP, UNIFORM_Q4_GROUP_SIZE,
};
use serde::Serialize;

/// Schema for the honest-roof receipt this module writes.
pub const HONEST_ROOF_SCHEMA: &str = "hawking.ascension.qwen38_honest_roof.v1";

/// Ledger-geometry active bytes (`theoretical_weight_bytes().active_bytes`).
/// GEMV payload + f32 norms + one gathered embed row. Embed table excluded.
pub const LEDGER_GEOMETRY_ACTIVE_BYTES: u64 = 13_618_141_856;

/// Bandwidth-receipt active bytes (`QWEN38_BANDWIDTH_BOUND.json`).
/// `total_payload (14_297_694_680) − embed_table (675_865_079)`.
pub const BANDWIDTH_RECEIPT_ACTIVE_BYTES: u64 = 13_621_829_601;

/// Source-constant / manifest-measured active bytes (`ACTIVE_BUDGET_BYTES`).
pub const SOURCE_CONSTANT_ACTIVE_BYTES: u64 = ACTIVE_BUDGET_BYTES;

/// Bytes the Q4 GEMV kernels actually stream per token (codes + f16 scales).
/// This is the defended denominator for `weight_addressing`.
pub const GEMV_PAYLOAD_BYTES: u64 = 13_611_663_360;

/// Sealed-ledger `weight_addressing.ns_per_token` (G024 / TOKEN_NS).
/// Isolated addr-probe fraction of class GEMVs, scale = 1.0.
pub const SEALED_WEIGHT_ADDRESSING_NS: f64 = 21_293_102.5;

/// Sealed-ledger median production GPU ns (`GPUEndTime − GPUStartTime`).
pub const SEALED_PRODUCTION_GPU_NS: u64 = 33_912_333;

/// Sealed-ledger median production wall ns (encode + submit + wait + tail).
pub const SEALED_PRODUCTION_WALL_NS: u64 = 35_227_917;

/// Published M3 Ultra peak. A datasheet number, not a measured decode roof.
pub const M3_ULTRA_PEAK_GB_S: f64 = 819.0;

/// HQ30UQ4 on-disk header for a rank-2 matrix: 32-byte prefix + 2 × u32 dims.
pub const HQ30UQ4_RANK2_HEADER_BYTES: u64 = 40;

/// Manifest class bytes from `QWEN38_ACTIVE_BUDGET_MEASURED.json`.
pub const MANIFEST_MLP_BYTES: u64 = 9_091_161_600;
pub const MANIFEST_LINEAR_BYTES: u64 = 2_961_704_064;
pub const MANIFEST_FULL_BYTES: u64 = 891_325_184;
pub const MANIFEST_LM_HEAD_BYTES: u64 = 675_430_440;
pub const MANIFEST_NORMS_BYTES: u64 = 2_642_952;
pub const MANIFEST_EMBED_TABLE_BYTES: u64 = 675_430_440;

/// Stale embed figure used by `QWEN38_BANDWIDTH_BOUND.json`.
pub const BANDWIDTH_RECEIPT_EMBED_BYTES: u64 = 675_865_079;

/// `total_payload` in that same receipt: manifest active + measured embed.
pub const BANDWIDTH_RECEIPT_TOTAL_PAYLOAD: u64 = 14_297_694_680;

#[derive(Clone, Debug, Serialize)]
pub struct ByteCountSource {
    pub name: &'static str,
    pub bytes: u64,
    pub what_it_counts: &'static str,
    pub why_wrong_for_weight_addressing: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ByteCountAdjudication {
    pub defended_bytes: u64,
    pub defended_name: &'static str,
    pub defended_reason: &'static str,
    pub gemv_payload_breakdown: GemvPayloadBreakdown,
    pub sources: Vec<ByteCountSource>,
    pub header_and_extra_accounting: HeaderAccounting,
}

#[derive(Clone, Debug, Serialize)]
pub struct GemvPayloadBreakdown {
    pub mlp_bytes: u64,
    pub linear_attn_bytes: u64,
    pub full_attn_bytes: u64,
    pub lm_head_bytes: u64,
    pub gemv_payload: u64,
    pub norms_bytes: u64,
    pub embed_row_bytes: u64,
    pub geometry_active: u64,
    pub embed_table_excluded: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HeaderAccounting {
    pub hq30uq4_rank2_header_bytes: u64,
    pub mlp_tensors: u64,
    pub linear_gemv_tensors: u64,
    pub full_gemv_tensors: u64,
    pub lm_head_tensors: u64,
    pub mlp_manifest_minus_geometry: u64,
    pub mlp_explained_as_headers: u64,
    pub lm_head_manifest_minus_geometry: u64,
    pub linear_manifest_minus_geometry: u64,
    pub linear_headers: u64,
    pub linear_extra_non_gemv: u64,
    pub full_manifest_minus_geometry: u64,
    pub full_headers: u64,
    pub full_extra_non_gemv: u64,
    pub note: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct DenominatorCorrection {
    pub wrong_97p6: Wrong97p6,
    pub correct_attribution: CorrectAttribution,
}

#[derive(Clone, Debug, Serialize)]
pub struct Wrong97p6 {
    pub claimed_pct_of_411p51: f64,
    pub bytes_used: u64,
    pub time_used_ns: u64,
    pub time_used_name: &'static str,
    pub achieved_gb_s: f64,
    pub ceiling_used_gb_s: f64,
    pub why_bytes_wrong: &'static str,
    pub why_time_wrong: &'static str,
    pub why_ceiling_wrong: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct CorrectAttribution {
    pub bytes: u64,
    pub bytes_name: &'static str,
    pub time_ns: f64,
    pub time_name: &'static str,
    pub achieved_gb_s: f64,
    pub time_is_weight_addressing_not_total_gpu: bool,
    pub sealed_probe_split: &'static str,
}

pub fn gemv_payload_from_geometry() -> GemvPayloadBreakdown {
    let w = theoretical_weight_bytes();
    let gemv = w.mlp_bytes + w.linear_attn_bytes + w.full_attn_bytes + w.lm_head_bytes;
    GemvPayloadBreakdown {
        mlp_bytes: w.mlp_bytes,
        linear_attn_bytes: w.linear_attn_bytes,
        full_attn_bytes: w.full_attn_bytes,
        lm_head_bytes: w.lm_head_bytes,
        gemv_payload: gemv,
        norms_bytes: w.norms_bytes,
        embed_row_bytes: w.embed_row_bytes,
        geometry_active: w.active_bytes,
        embed_table_excluded: w.embed_table_excluded_bytes,
    }
}

pub fn header_accounting(geo: &GemvPayloadBreakdown) -> HeaderAccounting {
    let mlp_tensors = (QWEN38_LAYERS as u64) * 3;
    let linear_gemv_tensors = (QWEN38_DELTANET_LAYERS as u64) * 3;
    let full_gemv_tensors = (QWEN38_GQA_LAYERS as u64) * 4;
    let lm_head_tensors = 1u64;
    let hdr = HQ30UQ4_RANK2_HEADER_BYTES;
    HeaderAccounting {
        hq30uq4_rank2_header_bytes: hdr,
        mlp_tensors,
        linear_gemv_tensors,
        full_gemv_tensors,
        lm_head_tensors,
        mlp_manifest_minus_geometry: MANIFEST_MLP_BYTES.saturating_sub(geo.mlp_bytes),
        mlp_explained_as_headers: mlp_tensors * hdr,
        lm_head_manifest_minus_geometry: MANIFEST_LM_HEAD_BYTES.saturating_sub(geo.lm_head_bytes),
        linear_manifest_minus_geometry: MANIFEST_LINEAR_BYTES.saturating_sub(geo.linear_attn_bytes),
        linear_headers: linear_gemv_tensors * hdr,
        linear_extra_non_gemv: MANIFEST_LINEAR_BYTES
            .saturating_sub(geo.linear_attn_bytes)
            .saturating_sub(linear_gemv_tensors * hdr),
        full_manifest_minus_geometry: MANIFEST_FULL_BYTES.saturating_sub(geo.full_attn_bytes),
        full_headers: full_gemv_tensors * hdr,
        full_extra_non_gemv: MANIFEST_FULL_BYTES
            .saturating_sub(geo.full_attn_bytes)
            .saturating_sub(full_gemv_tensors * hdr),
        note: "MLP and lm_head extras are exactly rank-2 HQ30UQ4 headers (40 B/tensor). Linear/full extras are headers plus mixer tensors the Q4 GEMV kernel never streams (conv, A_log, dt_bias, q/k RMS). Those bytes are not weight_addressing traffic.",
    }
}

pub fn adjudicate_byte_counts() -> ByteCountAdjudication {
    let geo = gemv_payload_from_geometry();
    let headers = header_accounting(&geo);
    ByteCountAdjudication {
        defended_bytes: geo.gemv_payload,
        defended_name: "geometry GEMV payload (codes + f16 scales)",
        defended_reason: "weight_addressing is the addr_probe fraction of the Q4 grouped GEMVs. The kernel loads 32 code bytes + 2 scale bytes per group of 64 weights. Headers stay on disk after load. Norms, embed-row gather, conv, and A_log are other components.",
        gemv_payload_breakdown: geo,
        sources: vec![
            ByteCountSource {
                name: "ledger_geometry_active",
                bytes: LEDGER_GEOMETRY_ACTIVE_BYTES,
                what_it_counts: "Q4 GEMV payload + f32 norms + one gathered embed row. Embed table excluded.",
                why_wrong_for_weight_addressing: "Includes 6_475_776 B of norms and 2_720 B of embed-row that the GEMV addr_probe does not load. Those live in `normalization` and `unattributed_residual`.",
            },
            ByteCountSource {
                name: "bandwidth_receipt_active",
                bytes: BANDWIDTH_RECEIPT_ACTIVE_BYTES,
                what_it_counts: "total_payload 14_297_694_680 minus a stale embed figure 675_865_079.",
                why_wrong_for_weight_addressing: "Arithmetic remainder, not a GEMV census. The embed subtraction is 434_639 B larger than the measured HQ30UQ4 embed table (675_430_440). total_payload itself is manifest-active + measured embed, so this number is a mis-classified remainder of the third source.",
            },
            ByteCountSource {
                name: "source_constant_ACTIVE_BUDGET_BYTES",
                bytes: SOURCE_CONSTANT_ACTIVE_BYTES,
                what_it_counts: "Sum of 755-tensor manifest classes excluding the embed table. Includes per-tensor HQ30UQ4 headers and extra mixer tensors.",
                why_wrong_for_weight_addressing: "Headers are not streamed by geo_tpr64. Extra linear/full bytes are not GEMV traffic. Manifest norms (2_642_952) also are not GEMV traffic.",
            },
        ],
        header_and_extra_accounting: headers,
    }
}

pub fn denominator_correction() -> DenominatorCorrection {
    let bytes = GEMV_PAYLOAD_BYTES;
    let wrong_bytes = LEDGER_GEOMETRY_ACTIVE_BYTES;
    let wrong_time = SEALED_PRODUCTION_GPU_NS;
    let wrong_gb = (wrong_bytes as f64) / (wrong_time as f64);
    let claimed = wrong_gb / HONEST_DECODE_CEILING_GB_S;
    let correct_gb = (bytes as f64) / SEALED_WEIGHT_ADDRESSING_NS;
    DenominatorCorrection {
        wrong_97p6: Wrong97p6 {
            claimed_pct_of_411p51: claimed,
            bytes_used: wrong_bytes,
            time_used_ns: wrong_time,
            time_used_name: "median production GPU ns (entire token, all kernels)",
            achieved_gb_s: wrong_gb,
            ceiling_used_gb_s: HONEST_DECODE_CEILING_GB_S,
            why_bytes_wrong: "13_618_141_856 is geometry-active, not GEMV payload. Norms + embed-row do not move in weight_addressing.",
            why_time_wrong: "33.91 ms includes DeltaNet, GQA, norms, SwiGLU, KV, terminal FMA, and residual GPU work that does not stream the GEMV bytes.",
            why_ceiling_wrong: "411.51 GB/s is the 512 MiB point of a sequential unique_once read-reduce. The 1024 MiB point of the same sweep was 301.63 GB/s and was discarded. unique_once cannot even take a 13.6 GB nbytes: the kernel's nbytes is uint32.",
        },
        correct_attribution: CorrectAttribution {
            bytes,
            bytes_name: "geometry GEMV payload (mlp+linear+full+lm_head codes+scales)",
            time_ns: SEALED_WEIGHT_ADDRESSING_NS,
            time_name: "sealed ledger weight_addressing ns (addr_probe fraction of isolated class GEMVs)",
            achieved_gb_s: correct_gb,
            time_is_weight_addressing_not_total_gpu: true,
            sealed_probe_split: "mlp addr 87.17% of GEMV; dn 90.51%; gqa 83.03%; lm_head 91.57%. Addressing dominates GEMV time; decode+FMA are the remainder.",
        },
    }
}

pub fn production_gemv_shapes() -> Vec<GemvShape> {
    let mut shapes = Vec::new();
    for _ in 0..QWEN38_LAYERS {
        shapes.push(GemvShape::new(QWEN38_INTERMEDIATE as u32, QWEN38_HIDDEN as u32, "mlp_gate"));
        shapes.push(GemvShape::new(QWEN38_INTERMEDIATE as u32, QWEN38_HIDDEN as u32, "mlp_up"));
        shapes.push(GemvShape::new(QWEN38_HIDDEN as u32, QWEN38_INTERMEDIATE as u32, "mlp_down"));
    }
    for _ in 0..QWEN38_DELTANET_LAYERS {
        shapes.push(GemvShape::new(QWEN38_QKVZ_ROWS as u32, QWEN38_HIDDEN as u32, "dn_qkvz"));
        shapes.push(GemvShape::new(QWEN38_BA_ROWS as u32, QWEN38_HIDDEN as u32, "dn_ba"));
        shapes.push(GemvShape::new(QWEN38_HIDDEN as u32, QWEN38_O_PROJ_COLS as u32, "dn_out"));
    }
    for _ in 0..QWEN38_GQA_LAYERS {
        shapes.push(GemvShape::new(QWEN38_Q_PROJ_ROWS as u32, QWEN38_HIDDEN as u32, "gqa_q"));
        shapes.push(GemvShape::new(QWEN38_KV_PROJ_ROWS as u32, QWEN38_HIDDEN as u32, "gqa_k"));
        shapes.push(GemvShape::new(QWEN38_KV_PROJ_ROWS as u32, QWEN38_HIDDEN as u32, "gqa_v"));
        shapes.push(GemvShape::new(QWEN38_HIDDEN as u32, QWEN38_O_PROJ_COLS as u32, "gqa_o"));
    }
    shapes.push(GemvShape::new(QWEN38_VOCAB as u32, QWEN38_HIDDEN as u32, "lm_head"));
    shapes
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct GemvShape {
    pub rows: u32,
    pub cols: u32,
    pub groups_per_row: u32,
    pub code_bytes: u64,
    pub scale_bytes: u64,
    pub payload_bytes: u64,
    pub class: &'static str,
}

impl GemvShape {
    pub fn new(rows: u32, cols: u32, class: &'static str) -> Self {
        let gpr = cols.div_ceil(UNIFORM_Q4_GROUP_SIZE as u32);
        let groups = (rows as u64) * (gpr as u64);
        let code = groups * (UNIFORM_Q4_CODE_BYTES_PER_GROUP as u64);
        let scale = groups * 2;
        Self {
            rows,
            cols,
            groups_per_row: gpr,
            code_bytes: code,
            scale_bytes: scale,
            payload_bytes: code + scale,
            class,
        }
    }

    pub fn launch_geo_tpr64_tg128(self) -> ((u32, u32, u32), (u32, u32, u32)) {
        let tg = 128u32;
        let grid = self.rows.div_ceil(2).saturating_mul(tg).max(tg);
        ((grid, 1, 1), (tg, 1, 1))
    }
}

pub fn q4_bytes_per_row(cols: u32) -> u64 {
    let gpr = cols.div_ceil(UNIFORM_Q4_GROUP_SIZE as u32) as u64;
    gpr * (UNIFORM_Q4_CODE_BYTES_PER_GROUP as u64 + 2)
}

pub fn rows_for_payload(cols: u32, payload: u64) -> u32 {
    let bpr = q4_bytes_per_row(cols);
    if bpr == 0 {
        return 0;
    }
    let rows = payload / bpr;
    rows.max(2) as u32
}

pub fn gb_s(bytes: u64, ns: u64) -> f64 {
    if ns == 0 {
        0.0
    } else {
        bytes as f64 / ns as f64
    }
}

pub fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut s = values.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

pub const Q4_FULL_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128";
pub const Q4_ADDR_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_addr_probe";
pub const Q4_DECODE_KERNEL: &str = "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_decode_probe";
pub const UNIQUE_ONCE_KERNEL: &str = "q80_decode_shape_unique_once";

/// Working-set sizes for the curve. Includes the discarded 1024 MiB unique_once
/// point and the production GEMV payload. Values are **payload bytes**.
pub fn sweep_payloads() -> Vec<SweepPoint> {
    const MIB: u64 = 1024 * 1024;
    vec![
        SweepPoint {
            label: "64mib",
            payload_bytes: 64 * MIB,
        },
        SweepPoint {
            label: "128mib",
            payload_bytes: 128 * MIB,
        },
        SweepPoint {
            label: "256mib",
            payload_bytes: 256 * MIB,
        },
        SweepPoint {
            label: "512mib",
            payload_bytes: 512 * MIB,
        },
        SweepPoint {
            label: "1024mib",
            payload_bytes: 1024 * MIB,
        },
        SweepPoint {
            label: "2048mib",
            payload_bytes: 2048 * MIB,
        },
        SweepPoint {
            label: "4096mib",
            payload_bytes: 4096 * MIB,
        },
        SweepPoint {
            label: "8192mib",
            payload_bytes: 8192 * MIB,
        },
        SweepPoint {
            label: "gemv_payload_13p612gb",
            payload_bytes: GEMV_PAYLOAD_BYTES,
        },
    ]
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SweepPoint {
    pub label: &'static str,
    pub payload_bytes: u64,
}

#[cfg(target_os = "macos")]
mod gpu {
    use super::*;
    use crate::metal::{MetalBatchTiming, MetalContext, MetalDispatchTiming};
    use crate::{Error, Result};
    use serde_json::{json, Value};
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    const PAGE: usize = 4096;
    const WARMUP: usize = 2;
    const TIMED: usize = 5;
    const UNIQUE_THREADS: u32 = 256 * 60;
    const UNIQUE_TG: u32 = 256;
    /// unique_once `nbytes` is uint32. Stay under 2 GiB per dispatch.
    const UNIQUE_CHUNK: u64 = 512 * 1024 * 1024;

    pub const DEFAULT_RECEIPT: &str =
        "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.json";
    pub const REDUCED_RECEIPT: &str =
        "receipts/ascent-2026-08-16/HONEST_ROOF_WEIGHT_ADDRESSING.reduced.json";

    pub(super) fn workspace_receipt(rel: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel)
    }

    fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        enc.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn require_dispatch_gpu(t: MetalDispatchTiming, label: &str) -> Result<u64> {
        match (t.gpu_start_ns, t.gpu_end_ns) {
            (Some(s), Some(e)) if e > s => Ok(e - s),
            _ => Err(Error::Metal(format!(
                "{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable"
            ))),
        }
    }

    fn require_batch_gpu(t: MetalBatchTiming, label: &str) -> Result<u64> {
        match (t.gpu_start_ns, t.gpu_end_ns) {
            (Some(s), Some(e)) if e > s => Ok(e - s),
            _ => Err(Error::Metal(format!(
                "{label}: MTLCommandBuffer GPUStartTime/GPUEndTime unavailable"
            ))),
        }
    }

    fn touch_pages(buf: &metal::Buffer, len: usize) {
        if len == 0 {
            return;
        }
        unsafe {
            let p = buf.contents() as *mut u8;
            if p.is_null() {
                return;
            }
            let mut off = 0;
            while off < len {
                *p.add(off) = 0x5a;
                off += PAGE;
            }
            *p.add(len - 1) = 0xa5;
        }
    }

    fn spread_json(ns: &[u64], bytes: u64) -> Value {
        let med = median_u64(ns);
        json!({
            "all_ns": ns,
            "median_ns": med,
            "min_ns": ns.iter().copied().min(),
            "max_ns": ns.iter().copied().max(),
            "median_gb_s": gb_s(bytes, med),
            "min_gb_s": ns.iter().copied().map(|t| gb_s(bytes, t)).fold(f64::INFINITY, f64::min),
            "max_gb_s": ns.iter().copied().map(|t| gb_s(bytes, t)).fold(f64::NEG_INFINITY, f64::max),
        })
    }

    struct Slab {
        codes: metal::Buffer,
        scales: metal::Buffer,
        input: metal::Buffer,
        output: metal::Buffer,
        unique: metal::Buffer,
        unique_out: metal::Buffer,
        max_payload: u64,
        max_cols: u32,
        max_rows: u32,
    }

    impl Slab {
        fn allocate(ctx: &MetalContext, max_payload: u64, max_cols: u32, max_rows: u32) -> Result<Self> {
            let gpr = max_cols.div_ceil(UNIFORM_Q4_GROUP_SIZE as u32) as u64;
            let max_groups = (max_rows as u64) * gpr;
            let code_len = (max_groups * UNIFORM_Q4_CODE_BYTES_PER_GROUP as u64) as usize;
            let scale_len = (max_groups * 2) as usize;
            let in_len = (max_cols as usize) * 4;
            let out_len = (max_rows as usize) * 4;
            let unique_len = max_payload as usize;
            let unique_out_len = (UNIQUE_THREADS as usize) * 4;

            let codes = ctx.new_buffer_checked(code_len.max(64))?;
            let scales = ctx.new_buffer_checked(scale_len.max(64))?;
            let input = ctx.new_buffer_checked(in_len.max(64))?;
            let output = ctx.new_buffer_checked(out_len.max(64))?;
            let unique = ctx.new_buffer_checked(unique_len.max(64))?;
            let unique_out = ctx.new_buffer_checked(unique_out_len.max(64))?;

            touch_pages(&codes, code_len.max(64));
            touch_pages(&scales, scale_len.max(64));
            touch_pages(&input, in_len.max(64));
            touch_pages(&output, out_len.max(64));
            touch_pages(&unique, unique_len.max(64));
            touch_pages(&unique_out, unique_out_len.max(64));

            // Keep a live store so the compiler/driver cannot treat the
            // unique slab as unread-only host memory.
            unsafe {
                let p = unique.contents() as *mut u8;
                if !p.is_null() && unique_len > 16 {
                    std::ptr::write(p as *mut u32, 0x3f80_0001);
                }
            }

            Ok(Self {
                codes,
                scales,
                input,
                output,
                unique,
                unique_out,
                max_payload,
                max_cols,
                max_rows,
            })
        }
    }

    fn time_q4_single(
        ctx: &MetalContext,
        slab: &Slab,
        kernel: &str,
        shape: GemvShape,
        reps: usize,
    ) -> Result<Vec<u64>> {
        if shape.rows > slab.max_rows || shape.cols > slab.max_cols {
            return Err(Error::Metal(format!(
                "shape {}x{} exceeds slab {}x{}",
                shape.rows, shape.cols, slab.max_rows, slab.max_cols
            )));
        }
        let (grid, tg) = shape.launch_geo_tpr64_tg128();
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = ctx.dispatch_threads_timed(kernel, grid, tg, |enc| {
                enc.set_buffer(0, Some(&slab.codes), 0);
                enc.set_buffer(1, Some(&slab.scales), 0);
                enc.set_buffer(2, Some(&slab.input), 0);
                enc.set_buffer(3, Some(&slab.output), 0);
                set_u32(enc, 4, shape.rows);
                set_u32(enc, 5, shape.cols);
                set_u32(enc, 6, shape.groups_per_row);
            })?;
            times.push(require_dispatch_gpu(t, kernel)?);
        }
        Ok(times)
    }

    fn time_q4_catalog(
        ctx: &MetalContext,
        slab: &Slab,
        kernel: &str,
        shapes: &[GemvShape],
        reps: usize,
    ) -> Result<(Vec<u64>, u64, u64)> {
        let mut code_off = 0u64;
        let mut scale_off = 0u64;
        let mut bytes = 0u64;
        for s in shapes {
            code_off = code_off.saturating_add(s.code_bytes);
            scale_off = scale_off.saturating_add(s.scale_bytes);
            bytes = bytes.saturating_add(s.payload_bytes);
        }
        if code_off > slab.codes.length() || scale_off > slab.scales.length() {
            return Err(Error::Metal(format!(
                "catalog codes {code_off} / scales {scale_off} exceed slab"
            )));
        }
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = ctx.dispatch_batch_timed(|batch| {
                let mut c_off = 0u64;
                let mut s_off = 0u64;
                for shape in shapes {
                    let (grid, tg) = shape.launch_geo_tpr64_tg128();
                    batch.dispatch_threads(kernel, grid, tg, |enc| {
                        enc.set_buffer(0, Some(&slab.codes), c_off);
                        enc.set_buffer(1, Some(&slab.scales), s_off);
                        enc.set_buffer(2, Some(&slab.input), 0);
                        enc.set_buffer(3, Some(&slab.output), 0);
                        set_u32(enc, 4, shape.rows);
                        set_u32(enc, 5, shape.cols);
                        set_u32(enc, 6, shape.groups_per_row);
                    })?;
                    c_off += shape.code_bytes;
                    s_off += shape.scale_bytes;
                }
                Ok(())
            })?;
            times.push(require_batch_gpu(t, kernel)?);
        }
        Ok((times, bytes, shapes.len() as u64))
    }

    fn time_unique_once(
        ctx: &MetalContext,
        slab: &Slab,
        nbytes: u64,
        reps: usize,
    ) -> Result<(Vec<u64>, u64)> {
        if nbytes > slab.max_payload {
            return Err(Error::Metal(format!(
                "unique_once {nbytes} exceeds slab {}",
                slab.max_payload
            )));
        }
        let chunks = if nbytes == 0 {
            0
        } else {
            nbytes.div_ceil(UNIQUE_CHUNK)
        };
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            if chunks <= 1 {
                let n = nbytes.min(u32::MAX as u64) as u32;
                let t = ctx.dispatch_threads_timed(
                    UNIQUE_ONCE_KERNEL,
                    (UNIQUE_THREADS, 1, 1),
                    (UNIQUE_TG, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&slab.unique), 0);
                        enc.set_buffer(1, Some(&slab.unique_out), 0);
                        set_u32(enc, 2, n);
                    },
                )?;
                times.push(require_dispatch_gpu(t, UNIQUE_ONCE_KERNEL)?);
            } else {
                let t = ctx.dispatch_batch_timed(|batch| {
                    let mut off = 0u64;
                    while off < nbytes {
                        let chunk = (nbytes - off).min(UNIQUE_CHUNK);
                        batch.dispatch_threads(
                            UNIQUE_ONCE_KERNEL,
                            (UNIQUE_THREADS, 1, 1),
                            (UNIQUE_TG, 1, 1),
                            |enc| {
                                enc.set_buffer(0, Some(&slab.unique), off);
                                enc.set_buffer(1, Some(&slab.unique_out), 0);
                                set_u32(enc, 2, chunk as u32);
                            },
                        )?;
                        off += chunk;
                    }
                    Ok(())
                })?;
                times.push(require_batch_gpu(t, UNIQUE_ONCE_KERNEL)?);
            }
        }
        Ok((times, chunks.max(1)))
    }

    fn measure_kernel_curve(
        ctx: &MetalContext,
        slab: &Slab,
        kernel: &str,
        cols: u32,
        points: &[SweepPoint],
        reduced: bool,
    ) -> Result<Vec<Value>> {
        let mut out = Vec::new();
        for p in points {
            if reduced && p.payload_bytes > 512 * 1024 * 1024 {
                continue;
            }
            let rows = rows_for_payload(cols, p.payload_bytes);
            if rows > slab.max_rows {
                continue;
            }
            let shape = GemvShape::new(rows, cols, "single_gemv");
            for _ in 0..WARMUP {
                let _ = time_q4_single(ctx, slab, kernel, shape, 1)?;
            }
            let ns = time_q4_single(ctx, slab, kernel, shape, TIMED)?;
            let med = median_u64(&ns);
            eprintln!(
                "  q4 {kernel} {} rows={rows} bytes={} median_ns={med} gb_s={:.3}",
                p.label,
                shape.payload_bytes,
                gb_s(shape.payload_bytes, med)
            );
            out.push(json!({
                "label": p.label,
                "kernel": kernel,
                "topology": "single_gemv",
                "rows": rows,
                "cols": cols,
                "payload_bytes": shape.payload_bytes,
                "requested_payload_bytes": p.payload_bytes,
                "dispatches": 1,
                "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
                "spread": spread_json(&ns, shape.payload_bytes),
            }));
        }
        Ok(out)
    }

    fn measure_unique_curve(
        ctx: &MetalContext,
        slab: &Slab,
        points: &[SweepPoint],
        reduced: bool,
    ) -> Result<Vec<Value>> {
        let mut out = Vec::new();
        for p in points {
            if reduced && p.payload_bytes > 1024 * 1024 * 1024 {
                continue;
            }
            if p.payload_bytes > slab.max_payload {
                continue;
            }
            for _ in 0..WARMUP {
                let _ = time_unique_once(ctx, slab, p.payload_bytes, 1)?;
            }
            let (ns, chunks) = time_unique_once(ctx, slab, p.payload_bytes, TIMED)?;
            let med = median_u64(&ns);
            eprintln!(
                "  unique_once {} bytes={} chunks={chunks} median_ns={med} gb_s={:.3}",
                p.label,
                p.payload_bytes,
                gb_s(p.payload_bytes, med)
            );
            out.push(json!({
                "label": p.label,
                "kernel": UNIQUE_ONCE_KERNEL,
                "topology": if chunks <= 1 { "single_dispatch" } else { "chunked_uint32_nbytes" },
                "payload_bytes": p.payload_bytes,
                "dispatches": chunks,
                "chunk_bytes": if chunks <= 1 { p.payload_bytes } else { UNIQUE_CHUNK },
                "threads": UNIQUE_THREADS,
                "why_chunked": "q80_decode_shape_unique_once takes constant uint nbytes; a single dispatch cannot cover 13.6 GB.",
                "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
                "spread": spread_json(&ns, p.payload_bytes),
            }));
        }
        Ok(out)
    }

    fn measure_tiled_organs(
        ctx: &MetalContext,
        slab: &Slab,
        kernel: &str,
        organ: GemvShape,
        target: u64,
        label: &str,
    ) -> Result<Value> {
        let n = (target / organ.payload_bytes).max(1);
        let shapes = vec![organ; n as usize];
        for _ in 0..WARMUP {
            let _ = time_q4_catalog(ctx, slab, kernel, &shapes, 1)?;
        }
        let (ns, bytes, disp) = time_q4_catalog(ctx, slab, kernel, &shapes, TIMED)?;
        let med = median_u64(&ns);
        eprintln!(
            "  tiled {kernel} {label} n={n} bytes={bytes} median_ns={med} gb_s={:.3}",
            gb_s(bytes, med)
        );
        Ok(json!({
            "label": label,
            "kernel": kernel,
            "topology": "tiled_production_organ",
            "organ_rows": organ.rows,
            "organ_cols": organ.cols,
            "organs": n,
            "payload_bytes": bytes,
            "dispatches": disp,
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
            "spread": spread_json(&ns, bytes),
        }))
    }

    fn measure_production_catalog(
        ctx: &MetalContext,
        slab: &Slab,
        kernel: &str,
    ) -> Result<Value> {
        let shapes = production_gemv_shapes();
        for _ in 0..WARMUP {
            let _ = time_q4_catalog(ctx, slab, kernel, &shapes, 1)?;
        }
        let (ns, bytes, disp) = time_q4_catalog(ctx, slab, kernel, &shapes, TIMED)?;
        let med = median_u64(&ns);
        eprintln!(
            "  catalog {kernel} disp={disp} bytes={bytes} median_ns={med} gb_s={:.3}",
            gb_s(bytes, med)
        );
        Ok(json!({
            "label": "production_catalog_401_gemvs",
            "kernel": kernel,
            "topology": "production_shape_catalog",
            "payload_bytes": bytes,
            "dispatches": disp,
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait",
            "spread": spread_json(&ns, bytes),
            "note": "401 GEMVs of the production organ mix (192 MLP + 144 DN + 64 GQA + 1 lm_head) on unique synthetic Q4 bytes. Same kernel and launch geometry as decode. Not a model-quality run.",
        }))
    }

    fn pick_median_gb(points: &[Value], label: &str) -> Option<f64> {
        points.iter().find_map(|p| {
            if p.get("label").and_then(|v| v.as_str()) == Some(label) {
                p.get("spread")
                    .and_then(|s| s.get("median_gb_s"))
                    .and_then(|v| v.as_f64())
            } else {
                None
            }
        })
    }

    fn verdict_from_sweep(
        unique: &[Value],
        q4_full: &[Value],
        q4_addr: &[Value],
        q4_decode: &[Value],
        catalog_full: Option<&Value>,
        catalog_addr: Option<&Value>,
    ) -> Value {
        let unique_512 = pick_median_gb(unique, "512mib");
        let unique_1024 = pick_median_gb(unique, "1024mib");
        let unique_full = pick_median_gb(unique, "gemv_payload_13p612gb");
        let addr_full = pick_median_gb(q4_addr, "gemv_payload_13p612gb");
        let full_full = pick_median_gb(q4_full, "gemv_payload_13p612gb");
        let dec_full = pick_median_gb(q4_decode, "gemv_payload_13p612gb");
        let cat_addr = catalog_addr
            .and_then(|v| v.get("spread"))
            .and_then(|s| s.get("median_gb_s"))
            .and_then(|v| v.as_f64());
        let cat_full = catalog_full
            .and_then(|v| v.get("spread"))
            .and_then(|s| s.get("median_gb_s"))
            .and_then(|v| v.as_f64());

        // Kernel roof = single unique-once Q4 GEMV addr_probe at 13.6 GB.
        // Catalog is the same bytes under production dispatch topology
        // (401 organs); it is a lower bound, not the kernel roof.
        let kernel_roof = addr_full.or(cat_addr);
        let topology_rate = cat_addr;
        let sealed_gb = (GEMV_PAYLOAD_BYTES as f64) / SEALED_WEIGHT_ADDRESSING_NS;
        let vs_roof = kernel_roof.map(|r| {
            if r > 0.0 {
                sealed_gb / r
            } else {
                0.0
            }
        });
        let alu_tax = match (addr_full, full_full) {
            (Some(a), Some(f)) if a > 0.0 => Some(1.0 - (f / a)),
            _ => None,
        };

        // Saturation: addressing is DRAM-saturated on this genome if the
        // sealed component sits near the measured Q4-addr roof AND the
        // full GEMV is not far behind addr (ALU is not the limiter).
        let near_roof = vs_roof.map(|f| f >= 0.90).unwrap_or(false);
        let alu_not_limiter = alu_tax.map(|t| t < 0.25).unwrap_or(false);
        let unique_is_not_ceiling = match (unique_512, kernel_roof) {
            (Some(u), Some(r)) => r > u * 1.10,
            _ => true,
        };

        let (saturated, resource, absence) = if near_roof && alu_not_limiter {
            (
                true,
                Some("unique-once DRAM traffic of the geo_tpr64_tg128 Q4 grouped-GEMV genome (codes + f16 scales). Not the unique_once sequential-reduce control, not the 819 GB/s datasheet."),
                None,
            )
        } else if !alu_not_limiter && alu_tax.unwrap_or(0.0) >= 0.25 {
            (
                false,
                None,
                Some("full GEMV is substantially slower than addr_probe; ALU/decode is a first-order term, not a 5-15% tax."),
            )
        } else if !near_roof {
            (
                false,
                None,
                Some("sealed weight_addressing is below the measured Q4-addr roof at 13.6 GB; addressing has unused bandwidth on this genome."),
            )
        } else {
            (false, None, Some("sweep did not produce a decisive roof comparison."))
        };

        json!({
            "saturated_on_this_genome": saturated,
            "named_resource": resource,
            "resource_absence": absence,
            "sealed_weight_addressing_gb_s": sealed_gb,
            "measured_q4_addr_kernel_roof_gb_s": kernel_roof,
            "measured_q4_addr_catalog_gb_s": topology_rate,
            "sealed_over_kernel_roof": vs_roof,
            "single_gemv_at_13p6gb": {
                "addr_gb_s": addr_full,
                "decode_gb_s": dec_full,
                "full_gb_s": full_full,
                "alu_plus_decode_tax_vs_addr": alu_tax,
            },
            "production_catalog_at_13p6gb": {
                "addr_gb_s": cat_addr,
                "full_gb_s": cat_full,
            },
            "unique_once_contrast": {
                "at_512mib_gb_s": unique_512,
                "at_1024mib_gb_s": unique_1024,
                "at_13p6gb_gb_s": unique_full,
                "is_not_the_q4_gemv_ceiling": unique_is_not_ceiling,
            },
            "published_peak_gb_s": M3_ULTRA_PEAK_GB_S,
            "refuted_ceiling_gb_s": HONEST_DECODE_CEILING_GB_S,
            "roof_is_conditioned_on": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128, cols=5120 groups, unique codes+scales, production launch (64 threads/row, 128-thread TG)",
        })
    }

    pub fn run_honest_roof_sweep(out_path: &Path, reduced: bool) -> Result<Value> {
        let started = Instant::now();
        let adj = adjudicate_byte_counts();
        let denom = denominator_correction();
        assert_eq!(adj.defended_bytes, GEMV_PAYLOAD_BYTES);
        assert_eq!(adj.gemv_payload_breakdown.gemv_payload, GEMV_PAYLOAD_BYTES);
        assert_eq!(
            adj.gemv_payload_breakdown.geometry_active,
            LEDGER_GEOMETRY_ACTIVE_BYTES
        );

        let ctx = MetalContext::new()?;
        let cols = QWEN38_HIDDEN as u32;
        let max_payload = if reduced {
            1024 * 1024 * 1024
        } else {
            GEMV_PAYLOAD_BYTES
        };
        let max_rows = rows_for_payload(cols, max_payload).max(if reduced {
            QWEN38_INTERMEDIATE as u32
        } else {
            QWEN38_VOCAB as u32
        });
        eprintln!(
            "honest_roof allocating slab payload={max_payload} rows={max_rows} cols={cols} reduced={reduced}"
        );
        let slab = Slab::allocate(&ctx, max_payload, cols, max_rows)?;
        let points = sweep_payloads();

        eprintln!("unique_once sweep (old control; not the roof)");
        let unique = measure_unique_curve(&ctx, &slab, &points, reduced)?;

        eprintln!("Q4 full GEMV sweep ({Q4_FULL_KERNEL})");
        let q4_full = measure_kernel_curve(&ctx, &slab, Q4_FULL_KERNEL, cols, &points, reduced)?;
        eprintln!("Q4 addr_probe sweep ({Q4_ADDR_KERNEL})");
        let q4_addr = measure_kernel_curve(&ctx, &slab, Q4_ADDR_KERNEL, cols, &points, reduced)?;
        eprintln!("Q4 decode_probe sweep ({Q4_DECODE_KERNEL})");
        let q4_decode = measure_kernel_curve(&ctx, &slab, Q4_DECODE_KERNEL, cols, &points, reduced)?;

        let gate = GemvShape::new(QWEN38_INTERMEDIATE as u32, QWEN38_HIDDEN as u32, "mlp_gate");
        let mut tiled = Vec::new();
        let tiled_targets: &[(u64, &str)] = if reduced {
            &[(512 * 1024 * 1024, "512mib_tiled_gate")]
        } else {
            &[
                (512 * 1024 * 1024, "512mib_tiled_gate"),
                (1024 * 1024 * 1024, "1024mib_tiled_gate"),
                (4096 * 1024 * 1024, "4096mib_tiled_gate"),
                (GEMV_PAYLOAD_BYTES, "13p612gb_tiled_gate"),
            ]
        };
        for (target, label) in tiled_targets {
            tiled.push(measure_tiled_organs(
                &ctx, &slab, Q4_FULL_KERNEL, gate, *target, label,
            )?);
            tiled.push(measure_tiled_organs(
                &ctx, &slab, Q4_ADDR_KERNEL, gate, *target, &format!("{label}_addr"),
            )?);
        }

        let (catalog_full, catalog_addr, catalog_decode) = if reduced {
            (None, None, None)
        } else {
            eprintln!("production catalog 401 GEMVs");
            (
                Some(measure_production_catalog(&ctx, &slab, Q4_FULL_KERNEL)?),
                Some(measure_production_catalog(&ctx, &slab, Q4_ADDR_KERNEL)?),
                Some(measure_production_catalog(&ctx, &slab, Q4_DECODE_KERNEL)?),
            )
        };

        let verdict = verdict_from_sweep(
            &unique,
            &q4_full,
            &q4_addr,
            &q4_decode,
            catalog_full.as_ref(),
            catalog_addr.as_ref(),
        );

        let commit = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .unwrap_or_default()
            .trim()
            .to_owned();

        let receipt = json!({
            "schema": HONEST_ROOF_SCHEMA,
            "date": "2026-08-17",
            "source_head_at_measurement": commit,
            "source_tree_clean_at_measurement": false,
            "timing_label": "GPU_PROTECTED_CPU_CONTENDED",
            "clean_box": false,
            "contamination_note": "GPU lane lock was held and timings are completed-command-buffer GPU timestamps, but concurrent CPU builds and a repeatedly respawned sealed-model supervisor were present. Unified-memory contention was not excluded; absolute roof values require a clean paired rerun.",
            "gpu_timestamp_authority": "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
            "reduced_sweep": reduced,
            "hardware": {
                "chip": "Apple M3 Ultra",
                "gpu_cores": 60,
                "unified_memory_bytes": 103_079_215_104u64,
                "published_peak_gb_s": M3_ULTRA_PEAK_GB_S,
            },
            "what_was_under_test": {
                "claimed_ceiling_gb_s": HONEST_DECODE_CEILING_GB_S,
                "claimed_pct": 0.976,
                "claimed_weight_addressing_ms": 21.293,
                "refuted_because": [
                    "411.51 is unique_once at 512 MiB, not Q4 grouped GEMV",
                    "the same unique_once sweep's 1024 MiB point is 301.63 GB/s and was discarded",
                    "97.6% used total production GPU time as the denominator",
                    "unique_once nbytes is uint32 so the old control cannot even name a 13.6 GB point without tiling",
                ],
            },
            "byte_count_adjudication": adj,
            "denominator_correction": denom,
            "unique_once_sweep": unique,
            "q4_single_gemv_full": q4_full,
            "q4_single_gemv_addr_probe": q4_addr,
            "q4_single_gemv_decode_probe": q4_decode,
            "q4_tiled_production_organ": tiled,
            "q4_production_catalog_full": catalog_full,
            "q4_production_catalog_addr_probe": catalog_addr,
            "q4_production_catalog_decode_probe": catalog_decode,
            "verdict": verdict,
            "sealed_ledger_cited_not_rerun": {
                "weight_addressing_ns": SEALED_WEIGHT_ADDRESSING_NS,
                "production_gpu_ns": SEALED_PRODUCTION_GPU_NS,
                "production_wall_ns": SEALED_PRODUCTION_WALL_NS,
                "source": "receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json",
            },
            "wall_s": started.elapsed().as_secs_f64(),
        });

        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::Metal(e.to_string()))?;
        }
        fs::write(out_path, serde_json::to_string_pretty(&receipt).unwrap() + "\n")
            .map_err(|e| Error::Metal(e.to_string()))?;
        eprintln!("honest_roof wrote {}", out_path.display());
        Ok(receipt)
    }

    pub fn run_default() -> Result<Value> {
        // Full sweep when HAWKING_HONEST_ROOF=1. The required
        // `cargo test … backend` filter runs a reduced curve so it stays
        // a test, not a 13.6 GB job.
        let full = std::env::var("HAWKING_HONEST_ROOF").ok().as_deref() == Some("1");
        let rel = if full { DEFAULT_RECEIPT } else { REDUCED_RECEIPT };
        run_honest_roof_sweep(&workspace_receipt(rel), !full)
    }
}

#[cfg(target_os = "macos")]
pub use gpu::{run_default, run_honest_roof_sweep, DEFAULT_RECEIPT};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen38_token_ns_ledger::{q4_matrix_bytes, EMBED_TABLE_BYTES};

    #[test]
    fn backend_honest_roof_byte_counts_adjudicated() {
        let adj = adjudicate_byte_counts();
        assert_eq!(adj.defended_bytes, GEMV_PAYLOAD_BYTES);
        assert_eq!(adj.gemv_payload_breakdown.mlp_bytes, 9_091_153_920);
        assert_eq!(adj.gemv_payload_breakdown.linear_attn_bytes, 2_953_789_440);
        assert_eq!(adj.gemv_payload_breakdown.full_attn_bytes, 891_289_600);
        assert_eq!(adj.gemv_payload_breakdown.lm_head_bytes, 675_430_400);
        assert_eq!(
            adj.gemv_payload_breakdown.geometry_active,
            LEDGER_GEOMETRY_ACTIVE_BYTES
        );
        assert_eq!(SOURCE_CONSTANT_ACTIVE_BYTES, 13_622_264_240);
        assert_eq!(BANDWIDTH_RECEIPT_ACTIVE_BYTES, 13_621_829_601);
        assert_eq!(
            BANDWIDTH_RECEIPT_TOTAL_PAYLOAD - BANDWIDTH_RECEIPT_EMBED_BYTES,
            BANDWIDTH_RECEIPT_ACTIVE_BYTES
        );
        assert_eq!(
            SOURCE_CONSTANT_ACTIVE_BYTES + MANIFEST_EMBED_TABLE_BYTES,
            BANDWIDTH_RECEIPT_TOTAL_PAYLOAD
        );
        assert_eq!(
            adj.header_and_extra_accounting.mlp_manifest_minus_geometry,
            adj.header_and_extra_accounting.mlp_explained_as_headers
        );
        assert_eq!(
            adj.header_and_extra_accounting.lm_head_manifest_minus_geometry,
            HQ30UQ4_RANK2_HEADER_BYTES
        );
        assert_eq!(EMBED_TABLE_BYTES, MANIFEST_EMBED_TABLE_BYTES);
        // Geometry q4_matrix_bytes is the payload the kernel streams.
        assert_eq!(
            q4_matrix_bytes(QWEN38_INTERMEDIATE as u64, QWEN38_HIDDEN as u64)
                + q4_matrix_bytes(QWEN38_INTERMEDIATE as u64, QWEN38_HIDDEN as u64)
                + q4_matrix_bytes(QWEN38_HIDDEN as u64, QWEN38_INTERMEDIATE as u64),
            adj.gemv_payload_breakdown.mlp_bytes / (QWEN38_LAYERS as u64)
        );
    }

    #[test]
    fn backend_honest_roof_97p6_is_wrong_denominator() {
        let d = denominator_correction();
        let pct = d.wrong_97p6.claimed_pct_of_411p51;
        assert!(
            (pct - 0.9758).abs() < 0.002,
            "reproduced 97.6% claim, got {pct}"
        );
        assert!(d.correct_attribution.time_is_weight_addressing_not_total_gpu);
        let gb = d.correct_attribution.achieved_gb_s;
        assert!(
            (gb - 639.25).abs() < 0.05,
            "sealed addressing GB/s should be 639.25, got {gb}"
        );
        assert!(gb > HONEST_DECODE_CEILING_GB_S);
        // A floor you beat is not a floor.
        let fake_floor_ns = (GEMV_PAYLOAD_BYTES as f64) / HONEST_DECODE_CEILING_GB_S;
        let over = SEALED_WEIGHT_ADDRESSING_NS / fake_floor_ns;
        assert!(over < 1.0, "measured_over_floor={over} must be < 1");
    }

    #[test]
    fn backend_honest_roof_production_catalog_payload_matches_geometry() {
        let shapes = production_gemv_shapes();
        assert_eq!(shapes.len(), 192 + 144 + 64 + 1);
        let sum: u64 = shapes.iter().map(|s| s.payload_bytes).sum();
        assert_eq!(sum, GEMV_PAYLOAD_BYTES);
        assert_eq!(
            q4_bytes_per_row(QWEN38_HIDDEN as u32),
            80 * (UNIFORM_Q4_CODE_BYTES_PER_GROUP as u64 + 2)
        );
        assert_eq!(rows_for_payload(QWEN38_HIDDEN as u32, GEMV_PAYLOAD_BYTES), 5_004_288);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn backend_honest_roof_gpu_sweep() {
        let reduced = std::env::var("HAWKING_HONEST_ROOF").ok().as_deref() != Some("1");
        let rel = if reduced {
            super::gpu::REDUCED_RECEIPT
        } else {
            super::gpu::DEFAULT_RECEIPT
        };
        let receipt = super::gpu::run_honest_roof_sweep(&super::gpu::workspace_receipt(rel), reduced)
            .expect("honest roof GPU sweep");
        assert_eq!(
            receipt.get("schema").and_then(|v| v.as_str()),
            Some(HONEST_ROOF_SCHEMA)
        );
        let unique = receipt
            .get("unique_once_sweep")
            .and_then(|v| v.as_array())
            .expect("unique_once curve");
        assert!(
            !unique.is_empty(),
            "unique_once sweep must report at least one point"
        );
        let q4 = receipt
            .get("q4_single_gemv_addr_probe")
            .and_then(|v| v.as_array())
            .expect("q4 addr curve");
        assert!(!q4.is_empty(), "Q4 addr_probe sweep must report points");
        for pt in q4 {
            let ns = pt
                .get("spread")
                .and_then(|s| s.get("median_ns"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            assert!(ns > 0, "GPU timestamp missing on {:?}", pt.get("label"));
        }
    }
}
