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

    fn bench_q4k_v2(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<Vec<f64>> {
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
                enc.set_bytes(
                    3,
                    size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
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

    fn bench_q3k(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<Vec<f64>> {
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
                enc.set_bytes(
                    3,
                    size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
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

    fn bench_f16_pinned(
        ctx: &MetalContext,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<Vec<f64>> {
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
                enc.set_bytes(
                    3,
                    size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
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

    pub fn run_kernel(
        kernel: &str,
        rows: usize,
        cols: usize,
        iterations: usize,
    ) -> Result<KernelBenchResult> {
        let ctx = MetalContext::new()?;
        let shape_str = format!("{rows}x{cols}");
        let tag = super::shape_tag(rows, cols);

        let samples = match kernel {
            "gemv_q4_k_m_v2_pinned_tcb" => bench_q4k_v2(&ctx, rows, cols, iterations)?,
            "gemv_q3_k_pinned_tcb" => bench_q3k(&ctx, rows, cols, iterations)?,
            "gemv_f16_metal_pinned" => bench_f16_pinned(&ctx, rows, cols, iterations)?,
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
pub fn run_kernel(
    kernel: &str,
    rows: usize,
    cols: usize,
    iterations: usize,
) -> Result<KernelBenchResult> {
    imp::run_kernel(kernel, rows, cols, iterations)
}
