//! Kernel-level micro-benchmarks for the `dismantle bench-kernel` subcommand.
//!
//! Allocates synthetic Metal buffers at production shapes, dispatches the
//! requested kernel N times, and reports per-dispatch timing statistics.
//! No model load required — measures GPU dispatch latency only.

use crate::Result;
use serde::{Deserialize, Serialize};

/// One timing result for a single (kernel, shape) combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelBenchResult {
    pub kernel: String,
    pub shape: String,
    pub shape_tag: String,
    pub iterations: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub commit: String,
    pub timestamp: String,
}

/// Production shapes used by V2-Lite and Mixtral in their decode paths.
/// Each entry: (rows, cols, shape_tag).
pub const V2_LITE_SHAPES: &[(usize, usize, &str)] = &[
    (1408, 2048, "v2_lite_gate_up"),
    (2048, 1408, "v2_lite_down"),
    (4096, 2048, "v2_lite_dense"),
    (2048, 2048, "v2_lite_attn_proj"),
    (102400, 2048, "v2_lite_lm_head"),
];

pub const MIXTRAL_SHAPES: &[(usize, usize, &str)] = &[
    (14336, 4096, "mixtral_gate_up"),
    (4096, 14336, "mixtral_down"),
    (4096, 4096, "mixtral_q_proj"),
    (1024, 4096, "mixtral_kv_proj"),
    (32000, 4096, "mixtral_lm_head"),
];

/// All supported kernel names for --all mode.
pub const ALL_KERNEL_NAMES: &[&str] = &[
    "gemv_q4_k_m_v2_pinned_tcb",
    "gemv_q3_k_pinned_tcb",
    "gemv_f16_metal_pinned",
    "gemv_q4_k_m_v3_xtg_pinned",
    "gemv_q4_k_m_v3_xtg_sumy_pinned",
    "moe_expert_pair_chained",
    "moe_expert_pair_fused",
    "mla_decode_kernel_fc_kbatch",
];

/// Parse a "ROWSxCOLS" string into (rows, cols).
pub fn parse_shape(s: &str) -> Result<(usize, usize)> {
    let parts: Vec<&str> = s.splitn(2, 'x').collect();
    if parts.len() != 2 {
        return Err(crate::Error::Model(format!(
            "invalid shape {s:?}: expected ROWSxCOLS (e.g. 1408x2048)"
        )));
    }
    let rows = parts[0]
        .parse::<usize>()
        .map_err(|e| crate::Error::Model(format!("invalid rows in shape {s:?}: {e}")))?;
    let cols = parts[1]
        .parse::<usize>()
        .map_err(|e| crate::Error::Model(format!("invalid cols in shape {s:?}: {e}")))?;
    if rows == 0 || cols == 0 {
        return Err(crate::Error::Model(format!(
            "shape {s:?}: rows and cols must be > 0"
        )));
    }
    Ok((rows, cols))
}

/// Find the canonical shape tag for a given (rows, cols).
pub fn shape_tag(rows: usize, cols: usize) -> String {
    for &(r, c, tag) in V2_LITE_SHAPES.iter().chain(MIXTRAL_SHAPES.iter()) {
        if r == rows && c == cols {
            return tag.to_string();
        }
    }
    format!("custom_{rows}x{cols}")
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use crate::metal::MetalContext;
    use std::mem::size_of;
    use std::time::Instant;

    fn now_iso() -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let s = secs % 60;
        let m = (secs / 60) % 60;
        let h = (secs / 3600) % 24;
        let days = secs / 86400;
        let year = 1970 + days / 365;
        let doy = days % 365;
        let month = doy / 30 + 1;
        let day = doy % 30 + 1;
        format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
    }

    fn git_short_head() -> String {
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into())
    }

    fn compute_stats(mut samples: Vec<f64>) -> (f64, f64, f64, f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let p50 = samples[n / 2];
        let p99_idx = ((n as f64 * 0.99) as usize).min(n - 1);
        let p99 = samples[p99_idx];
        let min = samples[0];
        let max = samples[n - 1];
        (mean, p50, p99, min, max)
    }

    // ── Q4_K_M v2 (gemm_q4_k_m_fused_v2) ────────────────────────────────────

    fn bench_q4k_v2(ctx: &MetalContext, rows: usize, cols: usize, iterations: usize) -> Result<Vec<f64>> {
        if cols % 256 != 0 {
            return Err(crate::Error::Kernel(format!(
                "gemv_q4_k_m_v2_pinned_tcb requires cols%256==0; got cols={cols}"
            )));
        }
        let blocks_per_row = cols / 256;
        let w_bytes = rows * blocks_per_row * 144;
        let x_bytes = cols * size_of::<f32>();
        let out_bytes = rows * size_of::<f32>();

        let w_buf = ctx.new_buffer(w_bytes);
        let x_buf = ctx.new_buffer(x_bytes);
        let out_buf = ctx.new_buffer(out_bytes);

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = (rows as u32 + 7) / 8;
        let grid = (n_tg * V2_TG, 1, 1);
        let tg = (V2_TG, 1, 1);

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads("gemm_q4_k_m_fused_v2", grid, tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(3, size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(4, size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
            })
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    // ── Q4_K_M v3_xtg (gemm_q4_k_m_v3_xtg) ──────────────────────────────────
    //
    // path-to-125 L7.1 cooperative-x_cache standalone GEMV. Same math as
    // v3_8r; loads x into threadgroup SRAM once per TG. 256-thread TG,
    // 8 simdgroups × 1 row each, cols×4 bytes of shmem.

    fn bench_q4k_v3_xtg(ctx: &MetalContext, rows: usize, cols: usize, iterations: usize) -> Result<Vec<f64>> {
        if cols % 256 != 0 {
            return Err(crate::Error::Kernel(format!(
                "gemv_q4_k_m_v3_xtg_pinned requires cols%256==0; got cols={cols}"
            )));
        }
        let blocks_per_row = cols / 256;
        let w_bytes = rows * blocks_per_row * 144;
        let x_bytes = cols * size_of::<f32>();
        let out_bytes = rows * size_of::<f32>();

        let w_buf = ctx.new_buffer(w_bytes);
        let x_buf = ctx.new_buffer(x_bytes);
        let out_buf = ctx.new_buffer(out_bytes);

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg = (rows as u32 + 7) / 8;
        let grid = (n_tg * TG, 1, 1);
        let tg_dim = (TG, 1, 1);
        let shmem_bytes = (cols as u64) * size_of::<f32>() as u64;

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads("gemm_q4_k_m_v3_xtg", grid, tg_dim, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(3, size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(4, size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    // ── Q4_K_M v3_xtg_sumy (gemm_q4_k_m_v3_xtg_sumy) ────────────────────────
    //
    // path-to-150 L7 / Stage 0.5 — v3_xtg + min-correction sumy trick.
    // Same geometry / shmem as v3_xtg; differs only in the inner-loop
    // arithmetic. Fair comparison to v3_xtg via identical synthetic
    // buffers and identical dispatch shape.

    fn bench_q4k_v3_xtg_sumy(ctx: &MetalContext, rows: usize, cols: usize, iterations: usize) -> Result<Vec<f64>> {
        if cols % 256 != 0 {
            return Err(crate::Error::Kernel(format!(
                "gemv_q4_k_m_v3_xtg_sumy_pinned requires cols%256==0; got cols={cols}"
            )));
        }
        let blocks_per_row = cols / 256;
        let w_bytes = rows * blocks_per_row * 144;
        let x_bytes = cols * size_of::<f32>();
        let out_bytes = rows * size_of::<f32>();

        let w_buf = ctx.new_buffer(w_bytes);
        let x_buf = ctx.new_buffer(x_bytes);
        let out_buf = ctx.new_buffer(out_bytes);

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg = (rows as u32 + 7) / 8;
        let grid = (n_tg * TG, 1, 1);
        let tg_dim = (TG, 1, 1);
        let shmem_bytes = (cols as u64) * size_of::<f32>() as u64;

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads("gemm_q4_k_m_v3_xtg_sumy", grid, tg_dim, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(3, size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(4, size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    // ── MoE expert-pair: chained union pipeline (baseline) ──────────────────
    //
    // Two-kernel dispatch:
    //   moe_gate_up_union_v2t (gate + up + silu_mul fused) → routed_act
    //   moe_down_union_v2t (down)                            → routed_out
    //
    // Shape: rows = hidden_out (= hidden_in for V2-Lite), cols = routed_mid.
    // Synthetic routing: K=1, top_k=N, N routes each assigned to a distinct
    // expert. This is the K=1 (greedy-verify) regime — favorable for the
    // fused variant because there is no expert-reuse to lose.
    //
    // n_experts and N are hardcoded to V2-Lite numbers (64, 6).

    fn synth_q4k_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; n_blocks * 144];
        let mut s = seed;
        for b in 0..n_blocks {
            let off = b * 144;
            s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let d = (((s >> 33) as u8 as f32) / 255.0 - 0.5) * 0.1;
            s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let dmin = (((s >> 33) as u8 as f32) / 255.0 - 0.5) * 0.01;
            bytes[off..off + 2]
                .copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            bytes[off + 2..off + 4]
                .copy_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
            for i in 4..16 {
                s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                bytes[off + i] = ((s >> 33) as u8) & 0x3F;
            }
            for i in 16..144 {
                s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                bytes[off + i] = (s >> 33) as u8;
            }
        }
        bytes
    }

    fn synth_q8_0_bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut bytes = vec![0u8; n_blocks * 34];
        let mut s = seed;
        for b in 0..n_blocks {
            let off = b * 34;
            s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let d = (((s >> 33) as u8 as f32) / 255.0 - 0.5) * 0.05;
            bytes[off..off + 2]
                .copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for i in 0..32 {
                s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                bytes[off + 2 + i] = (s >> 33) as u8;
            }
        }
        bytes
    }

    // V2-Lite-shaped MoE bench fixture. Allocates per-expert weight bytes
    // for all n_experts then runs a K=1, N=n_routes dispatch (each route
    // selects a distinct expert).
    struct MoeFixture {
        w_buf: crate::metal::PinnedBuffer,
        per_k_x_buf: crate::metal::PinnedBuffer,
        seg_buf: crate::metal::PinnedBuffer,
        sorted_kidx_buf: crate::metal::PinnedBuffer,
        sorted_slot_buf: crate::metal::PinnedBuffer,
        route_ids_buf: crate::metal::PinnedBuffer,
        route_kk_buf: crate::metal::PinnedBuffer,
        routed_act_buf: crate::metal::PinnedBuffer,
        routed_out_buf: crate::metal::PinnedBuffer,
        hidden_in: u32,
        routed_mid: u32,
        hidden_out: u32,
        n_experts: u32,
        n_routes: u32,
        gate_offset: u64,
        up_offset: u64,
        down_offset: u64,
    }

    fn build_moe_fixture(
        ctx: &MetalContext,
        hidden_in: usize,
        routed_mid: usize,
        hidden_out: usize,
    ) -> Result<MoeFixture> {
        const N_EXPERTS: u32 = 64;
        // n_routes can be overridden via DISMANTLE_MOE_BENCH_n_routes env
        // for sweeping K=1 (6 routes) vs K=4 (24 routes) regimes without a
        // rebuild. Default = 6 (K=1 top_k=6, no expert overlap — favorable
        // case for fused per-route TG design).
        let n_routes: u32 = std::env::var("DISMANTLE_MOE_BENCH_n_routes")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n: &u32| n >= 1 && n <= N_EXPERTS)
            .unwrap_or(6);
        let n_routes_const: u32 = n_routes; // capture for buffer sizing
        if hidden_in % 256 != 0 {
            return Err(crate::Error::Kernel(format!(
                "MoE bench fixture requires hidden_in%256==0; got {hidden_in}"
            )));
        }
        if routed_mid % 32 != 0 {
            return Err(crate::Error::Kernel(format!(
                "MoE bench fixture requires routed_mid%32==0; got {routed_mid}"
            )));
        }

        let bpr_q4 = hidden_in / 256;
        let bpr_q8 = routed_mid / 32;
        let per_gate_bytes = routed_mid * bpr_q4 * 144;
        let per_up_bytes = per_gate_bytes;
        let per_down_bytes = hidden_out * bpr_q8 * 34;

        let total_gate_bytes = N_EXPERTS as usize * per_gate_bytes;
        let total_up_bytes = N_EXPERTS as usize * per_up_bytes;
        let total_down_bytes = N_EXPERTS as usize * per_down_bytes;

        let gate_offset = 0u64;
        let up_offset = total_gate_bytes as u64;
        let down_offset = up_offset + total_up_bytes as u64;

        let mut combined =
            Vec::with_capacity(total_gate_bytes + total_up_bytes + total_down_bytes);
        combined.extend(synth_q4k_bytes(N_EXPERTS as usize * routed_mid * bpr_q4, 0xAA00_AA00));
        combined.extend(synth_q4k_bytes(N_EXPERTS as usize * routed_mid * bpr_q4, 0xBB00_BB00));
        combined.extend(synth_q8_0_bytes(N_EXPERTS as usize * hidden_out * bpr_q8, 0xCC00_CC00));
        let w_buf = ctx.new_buffer_with_bytes(&combined);

        // K=1 query, hidden_in floats.
        let x_f32: Vec<f32> = (0..hidden_in)
            .map(|i| {
                let v = (i as u32).wrapping_mul(0x9E37_79B9) as i32 as f32;
                v / (i32::MAX as f32) * 0.5
            })
            .collect();
        let per_k_x_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x_f32));

        // Routing: route i ∈ [0, n_routes) → expert i, slot i (top_k=n_routes).
        // segment_starts[expert] = i if expert < n_routes else n_routes (sentinel).
        let union_n = n_routes;
        let mut segment_starts = vec![union_n; (N_EXPERTS + 1) as usize];
        for i in 0..n_routes {
            segment_starts[i as usize] = i;
        }
        // Last sentinel.
        segment_starts[N_EXPERTS as usize] = union_n;

        let sorted_kidx: Vec<u32> = vec![0u32; n_routes as usize]; // all from K=0
        let sorted_slot: Vec<u32> = (0..n_routes).collect();
        let route_ids: Vec<u32> = (0..n_routes).collect(); // route i → expert i
        let route_kk: Vec<u32> = vec![0u32; n_routes as usize];

        let seg_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&segment_starts));
        let sorted_kidx_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&sorted_kidx));
        let sorted_slot_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&sorted_slot));
        let route_ids_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&route_ids));
        let route_kk_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&route_kk));

        // Intermediate (only used by chained): K=1, top_k=n_routes, mid.
        // Layout matches moe_gate_up_union_v2t / moe_down_union_v2t:
        //   routed_act[(kk, slot), :]  → kk * top_k * routed_mid + slot * routed_mid + i
        // For our setup K=1, top_k=n_routes → routed_act size = n_routes * routed_mid.
        let routed_act_buf =
            ctx.new_buffer(n_routes as usize * routed_mid * size_of::<f32>());
        // Output buffer:
        //   chained:  routed_out[(kk, slot), :] (n_routes * hidden_out)
        //   fused:    routed_out[route, :]      (n_routes * hidden_out)
        let routed_out_buf =
            ctx.new_buffer(n_routes as usize * hidden_out * size_of::<f32>());

        Ok(MoeFixture {
            w_buf,
            per_k_x_buf,
            seg_buf,
            sorted_kidx_buf,
            sorted_slot_buf,
            route_ids_buf,
            route_kk_buf,
            routed_act_buf,
            routed_out_buf,
            hidden_in: hidden_in as u32,
            routed_mid: routed_mid as u32,
            hidden_out: hidden_out as u32,
            n_experts: N_EXPERTS,
            n_routes: n_routes,
            gate_offset,
            up_offset,
            down_offset,
        })
    }

    fn bench_moe_chained(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<Vec<f64>> {
        // rows = hidden_out (= hidden_in for V2-Lite), cols = routed_mid.
        let hidden_io = rows;
        let routed_mid = cols;
        let fx = build_moe_fixture(ctx, hidden_io, routed_mid, hidden_io)?;
        let k_batch: u32 = 1;
        let top_k: u32 = fx.n_routes;
        let union_n: u32 = fx.n_routes;

        let gate_off = fx.gate_offset;
        let up_off = fx.up_offset;
        let down_off = fx.down_offset;
        const TG: u32 = 256;
        let gate_up_n_tg_x = (fx.routed_mid + 7) / 8;
        let gate_up_grid = (gate_up_n_tg_x * TG, fx.n_experts, 1);
        let down_n_tg_x = (fx.hidden_out + 7) / 8;
        let down_grid = (down_n_tg_x * TG, fx.n_experts, 1);
        let tg_dim = (TG, 1, 1);
        let gate_up_shmem = (k_batch as u64) * (fx.hidden_in as u64) * size_of::<f32>() as u64;

        let hidden_in = fx.hidden_in;
        let routed_mid_u = fx.routed_mid;
        let hidden_out = fx.hidden_out;
        let n_experts = fx.n_experts;

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads("moe_gate_up_union_v2t", gate_up_grid, tg_dim, |enc| {
                enc.set_buffer(0, Some(&fx.w_buf), 0);
                enc.set_buffer(1, Some(&fx.seg_buf), 0);
                enc.set_buffer(2, Some(&fx.sorted_kidx_buf), 0);
                enc.set_buffer(3, Some(&fx.sorted_slot_buf), 0);
                enc.set_buffer(4, Some(&fx.per_k_x_buf), 0);
                enc.set_buffer(5, Some(&fx.routed_act_buf), 0);
                enc.set_bytes(6, size_of::<u64>() as u64, &gate_off as *const u64 as *const _);
                enc.set_bytes(7, size_of::<u64>() as u64, &up_off as *const u64 as *const _);
                enc.set_bytes(8, size_of::<u32>() as u64, &routed_mid_u as *const u32 as *const _);
                enc.set_bytes(9, size_of::<u32>() as u64, &hidden_in as *const u32 as *const _);
                enc.set_bytes(10, size_of::<u32>() as u64, &k_batch as *const u32 as *const _);
                enc.set_bytes(11, size_of::<u32>() as u64, &top_k as *const u32 as *const _);
                enc.set_bytes(12, size_of::<u32>() as u64, &n_experts as *const u32 as *const _);
                enc.set_bytes(13, size_of::<u32>() as u64, &union_n as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, gate_up_shmem);
            })?;
            ctx.dispatch_threads("moe_down_union_v2t", down_grid, tg_dim, |enc| {
                enc.set_buffer(0, Some(&fx.w_buf), 0);
                enc.set_buffer(1, Some(&fx.seg_buf), 0);
                enc.set_buffer(2, Some(&fx.sorted_kidx_buf), 0);
                enc.set_buffer(3, Some(&fx.sorted_slot_buf), 0);
                enc.set_buffer(4, Some(&fx.routed_act_buf), 0);
                enc.set_buffer(5, Some(&fx.routed_out_buf), 0);
                enc.set_bytes(6, size_of::<u64>() as u64, &down_off as *const u64 as *const _);
                enc.set_bytes(7, size_of::<u32>() as u64, &hidden_out as *const u32 as *const _);
                enc.set_bytes(8, size_of::<u32>() as u64, &routed_mid_u as *const u32 as *const _);
                enc.set_bytes(9, size_of::<u32>() as u64, &k_batch as *const u32 as *const _);
                enc.set_bytes(10, size_of::<u32>() as u64, &top_k as *const u32 as *const _);
                enc.set_bytes(11, size_of::<u32>() as u64, &n_experts as *const u32 as *const _);
                enc.set_bytes(12, size_of::<u32>() as u64, &union_n as *const u32 as *const _);
            })?;
            Ok(())
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    fn bench_moe_fused(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<Vec<f64>> {
        let hidden_io = rows;
        let routed_mid = cols;
        let fx = build_moe_fixture(ctx, hidden_io, routed_mid, hidden_io)?;

        let gate_off = fx.gate_offset;
        let up_off = fx.up_offset;
        let down_off = fx.down_offset;
        let hidden_in = fx.hidden_in;
        let routed_mid_u = fx.routed_mid;
        let hidden_out = fx.hidden_out;
        let n_routes = fx.n_routes;
        let n_experts = fx.n_experts;
        const TG: u32 = 256;
        let grid = (TG, n_routes, 1);
        let tg_dim = (TG, 1, 1);
        let shmem = (hidden_in as u64 + routed_mid_u as u64) * size_of::<f32>() as u64;

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads("moe_expert_pair_fused", grid, tg_dim, |enc| {
                enc.set_buffer(0, Some(&fx.w_buf), 0);
                enc.set_buffer(1, Some(&fx.route_ids_buf), 0);
                enc.set_buffer(2, Some(&fx.route_kk_buf), 0);
                enc.set_buffer(3, Some(&fx.per_k_x_buf), 0);
                enc.set_buffer(4, Some(&fx.routed_out_buf), 0);
                enc.set_bytes(5, size_of::<u64>() as u64, &gate_off as *const u64 as *const _);
                enc.set_bytes(6, size_of::<u64>() as u64, &up_off as *const u64 as *const _);
                enc.set_bytes(7, size_of::<u64>() as u64, &down_off as *const u64 as *const _);
                enc.set_bytes(8, size_of::<u32>() as u64, &hidden_in as *const u32 as *const _);
                enc.set_bytes(9, size_of::<u32>() as u64, &routed_mid_u as *const u32 as *const _);
                enc.set_bytes(10, size_of::<u32>() as u64, &hidden_out as *const u32 as *const _);
                enc.set_bytes(11, size_of::<u32>() as u64, &n_routes as *const u32 as *const _);
                enc.set_bytes(12, size_of::<u32>() as u64, &n_experts as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem);
            })
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    // ── Q3_K (gemm_q3_k_fused_v2) ────────────────────────────────────────────

    fn bench_q3k(ctx: &MetalContext, rows: usize, cols: usize, iterations: usize) -> Result<Vec<f64>> {
        if cols % 256 != 0 {
            return Err(crate::Error::Kernel(format!(
                "gemv_q3_k_pinned_tcb requires cols%256==0; got cols={cols}"
            )));
        }
        let blocks_per_row = cols / 256;
        let w_bytes = rows * blocks_per_row * 110;
        let x_bytes = cols * size_of::<f32>();
        let out_bytes = rows * size_of::<f32>();

        let w_buf = ctx.new_buffer(w_bytes);
        let x_buf = ctx.new_buffer(x_bytes);
        let out_buf = ctx.new_buffer(out_bytes);

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg = (rows as u32 + 7) / 8;
        let grid = (n_tg * TG, 1, 1);
        let tg_dim = (TG, 1, 1);

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads("gemm_q3_k_fused_v2", grid, tg_dim, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(3, size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(4, size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
            })
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    // ── F16 pinned (gemv_f16) ─────────────────────────────────────────────────

    fn bench_f16_pinned(ctx: &MetalContext, rows: usize, cols: usize, iterations: usize) -> Result<Vec<f64>> {
        let w_bytes = rows * cols * 2; // f16 = 2 bytes
        let x_bytes = cols * size_of::<f32>();
        let out_bytes = rows * size_of::<f32>();

        let w_buf = ctx.new_buffer(w_bytes);
        let x_buf = ctx.new_buffer(x_bytes);
        let out_buf = ctx.new_buffer(out_bytes);

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG_SIZE: u32 = 256;
        let shmem = (TG_SIZE as u64) * size_of::<f32>() as u64;
        let grid = (rows_u32 * TG_SIZE, 1, 1);
        let tg = (TG_SIZE, 1, 1);

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads("gemv_f16", grid, tg, |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(3, size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(4, size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem);
            })
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    // ── MLA kbatch (mla_decode_kernel_fc_kbatch) ────────────────────────────
    //
    // path-to-150 Phase E E.0.a — pre-validation K-sweep bench for the
    // K-batched MLA decode kernel. Answers: does attention-verifier cost
    // stay roughly flat as K grows? If yes, the tree-decode lever is
    // physical on V2-Lite; if no, Phase E is killed at the cheapest gate.
    //
    // V2-Lite production MLA shape is hardcoded (n_heads=128,
    // kv_lora_rank=512, qk_nope=128, qk_rope=64, v_head_dim=128). The
    // rows/cols CLI args are ignored. K and seq_len are overridable via
    // env vars DISMANTLE_MLA_BENCH_k (default 4) and
    // DISMANTLE_MLA_BENCH_seq_len (default 256), mirroring the MoE
    // fixture's env-override pattern. Kernel hard-caps k_batch ∈ [1, 8].
    //
    // Dispatch geometry mirrors mla_decode_metal_kbatch (parallel_k.rs:323):
    // grid = (n_heads × TG_SIZE, 1, 1), tg = (TG_SIZE, 1, 1), with TG_SIZE
    // = 256. Two threadgroup buffers (q_nope_proj_k, c_kv_wt_k), each
    // sized k_batch × kv_lora_rank × sizeof(f32).

    fn bench_mla_kbatch(ctx: &MetalContext, _rows: usize, _cols: usize, iterations: usize) -> Result<Vec<f64>> {
        // V2-Lite MLA shape constants (deepseek_v2.rs:80-107).
        const N_HEADS: u32 = 128;
        const KV_LORA_RANK: u32 = 512;
        const QK_NOPE_HEAD_DIM: u32 = 128;
        const QK_ROPE_HEAD_DIM: u32 = 64;
        const V_HEAD_DIM: u32 = 128;
        const TG_SIZE: u32 = 256;

        let k_batch: u32 = std::env::var("DISMANTLE_MLA_BENCH_k")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&k: &u32| (1..=8).contains(&k))
            .unwrap_or(4);
        let seq_len: u32 = std::env::var("DISMANTLE_MLA_BENCH_seq_len")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&s: &u32| s >= k_batch && s <= 8192)
            .unwrap_or(256);

        let q_head_dim = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
        let q_bytes =
            (k_batch as usize) * (N_HEADS as usize) * (q_head_dim as usize) * size_of::<f32>();
        let c_kv_bytes = (seq_len as usize) * (KV_LORA_RANK as usize) * size_of::<f32>();
        let k_pe_bytes = (seq_len as usize) * (QK_ROPE_HEAD_DIM as usize) * size_of::<f32>();
        let kv_b_bytes = (N_HEADS as usize)
            * ((QK_NOPE_HEAD_DIM + V_HEAD_DIM) as usize)
            * (KV_LORA_RANK as usize)
            * size_of::<f32>();
        let out_bytes =
            (k_batch as usize) * (N_HEADS as usize) * (V_HEAD_DIM as usize) * size_of::<f32>();
        let scores_bytes = (N_HEADS as usize)
            * (k_batch as usize)
            * (seq_len as usize)
            * size_of::<f32>();

        let q_buf = ctx.new_buffer(q_bytes);
        let c_kv_buf = ctx.new_buffer(c_kv_bytes);
        let k_pe_buf = ctx.new_buffer(k_pe_bytes);
        let kv_b_buf = ctx.new_buffer(kv_b_bytes);
        let out_buf = ctx.new_buffer(out_bytes);
        let scores_buf = ctx.new_buffer(scores_bytes);

        let scale: f32 = 1.0 / ((q_head_dim as f32).sqrt());
        let qp_bytes = (k_batch as u64) * (KV_LORA_RANK as u64) * size_of::<f32>() as u64;
        let cwt_bytes = (k_batch as u64) * (KV_LORA_RANK as u64) * size_of::<f32>() as u64;

        let dispatch = |ctx: &MetalContext| -> Result<()> {
            ctx.dispatch_threads(
                "mla_decode_kernel_fc_kbatch",
                (N_HEADS * TG_SIZE, 1, 1),
                (TG_SIZE, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q_buf), 0);
                    enc.set_buffer(1, Some(&c_kv_buf), 0);
                    enc.set_buffer(2, Some(&k_pe_buf), 0);
                    enc.set_buffer(3, Some(&kv_b_buf), 0);
                    enc.set_buffer(4, Some(&out_buf), 0);
                    enc.set_buffer(5, Some(&scores_buf), 0);
                    enc.set_bytes(6, size_of::<u32>() as u64, &N_HEADS as *const u32 as *const _);
                    enc.set_bytes(7, size_of::<u32>() as u64, &QK_NOPE_HEAD_DIM as *const u32 as *const _);
                    enc.set_bytes(8, size_of::<u32>() as u64, &QK_ROPE_HEAD_DIM as *const u32 as *const _);
                    enc.set_bytes(9, size_of::<u32>() as u64, &V_HEAD_DIM as *const u32 as *const _);
                    enc.set_bytes(10, size_of::<u32>() as u64, &KV_LORA_RANK as *const u32 as *const _);
                    enc.set_bytes(11, size_of::<u32>() as u64, &seq_len as *const u32 as *const _);
                    enc.set_bytes(12, size_of::<f32>() as u64, &scale as *const f32 as *const _);
                    enc.set_bytes(13, size_of::<u32>() as u64, &k_batch as *const u32 as *const _);
                    enc.set_threadgroup_memory_length(0, qp_bytes);
                    enc.set_threadgroup_memory_length(1, cwt_bytes);
                },
            )
        };

        for _ in 0..50 {
            dispatch(ctx)?;
        }

        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let t0 = Instant::now();
            dispatch(ctx)?;
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        Ok(samples)
    }

    pub fn run_kernel(kernel: &str, rows: usize, cols: usize, iterations: usize) -> Result<KernelBenchResult> {
        let ctx = MetalContext::new()?;
        let shape_str = format!("{rows}x{cols}");
        let tag = super::shape_tag(rows, cols);

        let samples = match kernel {
            "gemv_q4_k_m_v2_pinned_tcb" => bench_q4k_v2(&ctx, rows, cols, iterations)?,
            "gemv_q3_k_pinned_tcb" => bench_q3k(&ctx, rows, cols, iterations)?,
            "gemv_f16_metal_pinned" => bench_f16_pinned(&ctx, rows, cols, iterations)?,
            "gemv_q4_k_m_v3_xtg_pinned" => bench_q4k_v3_xtg(&ctx, rows, cols, iterations)?,
            "gemv_q4_k_m_v3_xtg_sumy_pinned" => bench_q4k_v3_xtg_sumy(&ctx, rows, cols, iterations)?,
            "moe_expert_pair_chained" => bench_moe_chained(&ctx, rows, cols, iterations)?,
            "moe_expert_pair_fused" => bench_moe_fused(&ctx, rows, cols, iterations)?,
            // V2-Lite MLA shape is hardcoded inside; rows/cols ignored.
            "mla_decode_kernel_fc_kbatch" => bench_mla_kbatch(&ctx, rows, cols, iterations)?,
            other => {
                return Err(crate::Error::Kernel(format!(
                    "unknown kernel {other:?}; supported: {}",
                    super::ALL_KERNEL_NAMES.join(", ")
                )))
            }
        };

        let (mean, p50, p99, min, max) = compute_stats(samples);
        Ok(KernelBenchResult {
            kernel: kernel.to_string(),
            shape: shape_str,
            shape_tag: tag,
            iterations,
            mean_us: mean,
            p50_us: p50,
            p99_us: p99,
            min_us: min,
            max_us: max,
            commit: git_short_head(),
            timestamp: now_iso(),
        })
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::*;

    pub fn run_kernel(
        kernel: &str,
        _rows: usize,
        _cols: usize,
        _iterations: usize,
    ) -> Result<KernelBenchResult> {
        Err(crate::Error::Kernel(format!(
            "bench-kernel requires macOS (Metal); kernel={kernel}"
        )))
    }
}

/// Bench a single named kernel at the given (rows, cols, iterations).
pub fn run_kernel(kernel: &str, rows: usize, cols: usize, iterations: usize) -> Result<KernelBenchResult> {
    imp::run_kernel(kernel, rows, cols, iterations)
}
