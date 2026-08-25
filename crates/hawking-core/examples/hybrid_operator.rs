//! N038 — binary bulk + distributed correction as ONE fused native operator.
//!
//! Two representations, dense_w=0, unique-weight 64-layer MLP token graph
//! (192 GEMVs), >=7 reps:
//!   binary_g64 + rank-r  (U, V f16; V^T x fused into the binary sweep)
//!   binary_g64 + K=2 shared-basis residual (signs amortized, scales per layer)
//!
//! Baselines (binary, q2f) reuse `bytes_frontier.metal`. COMPLETE_TOKEN_NS is
//! composed in Python with the N021 non-MLP residual.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example hybrid_operator
//! ./tools/gpu_lane_lock.sh n038-hybrid \
//!   workspace/ops/build/rust/release-fast/examples/hybrid_operator \
//!   --reps 7 --out receipts/headless/_HYBRID_OPERATOR_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

fn usage() -> &'static str {
    "usage: hybrid_operator [--reps N] [--warmup N] [--layers N] [--out FILE] [--skip-unique]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("hybrid_operator: {message}");
    process::exit(2);
}

struct Args {
    reps: usize,
    warmup: usize,
    layers: usize,
    skip_unique: bool,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut reps = 7usize;
    let mut warmup = 2usize;
    let mut layers = 64usize;
    let mut skip_unique = false;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--reps" => {
                reps = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--reps"));
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--warmup"));
            }
            "--layers" => {
                layers = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--layers"));
            }
            "--skip-unique" => skip_unique = true,
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if reps < 7 {
        fail("--reps must be >= 7");
    }
    Args {
        reps,
        warmup,
        layers: layers.max(1),
        skip_unique,
        out,
    }
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("hybrid_operator is Metal-only");
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run(parse_args());
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use half::f16;
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
        MTLSize,
    };

    const HYBRID_SHADER: &str = include_str!("../shaders/hybrid_operator.metal");
    const BASE_SHADER: &str = include_str!("../shaders/bytes_frontier.metal");
    const PASS_TOL: f32 = 3e-2;
    const GATE_ROWS: u32 = 17408;
    const GATE_COLS: u32 = 5120;
    const DOWN_ROWS: u32 = 5120;
    const DOWN_COLS: u32 = 17408;
    const GROUP: u32 = 64;
    const N021_COMPLETE_GPU_NS: u64 = 27_547_874;
    const N032_Q2F_MLP_GPU_NS: u64 = 15_738_249;
    const MLP_ELEMENTS: u64 = 17_112_760_320;
    const PARENT_PARAMS: u64 = 26_895_998_464;
    const Q4_ATTN_F32_BYTES: u64 = 5_206_533_080;

    fn align256(n: usize) -> usize {
        (n + 255) & !255
    }

    fn as_u8_u16(v: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
    }
    fn as_u8_f32(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        enc.set_bytes(index, 4, &value as *const u32 as *const _);
    }

    fn gpu_ns(cmd: &metal::CommandBufferRef) -> Option<u64> {
        let start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
        let end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
        if end > start && start > 0.0 {
            Some(((end - start) * 1e9) as u64)
        } else {
            None
        }
    }

    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn median_u64(mut v: Vec<u64>) -> Option<u64> {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    }

    fn spread(values: &[u64]) -> Value {
        if values.is_empty() {
            return json!({"n": 0, "min": null, "median": null, "max": null, "all": []});
        }
        let mut s = values.to_vec();
        s.sort_unstable();
        json!({
            "n": s.len(),
            "min": s.first().copied(),
            "median": s[s.len() / 2],
            "max": s.last().copied(),
            "all": values,
        })
    }

    fn ranges_overlap(a: &[u64], b: &[u64]) -> bool {
        let (Some(&amin), Some(&amax), Some(&bmin), Some(&bmax)) =
            (a.iter().min(), a.iter().max(), b.iter().min(), b.iter().max())
        else {
            return false;
        };
        amin <= bmax && bmin <= amax
    }

    fn new_buf(device: &Device, bytes: &[u8]) -> Buffer {
        device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn new_empty(device: &Device, bytes: usize) -> Buffer {
        device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
    }

    fn fill_u8(n: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![0u8; n];
        let mut s = seed | 1;
        for chunk in v.chunks_mut(8) {
            s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let b = s.to_le_bytes();
            let n = chunk.len();
            chunk.copy_from_slice(&b[..n]);
        }
        v
    }

    fn fill_u16(n: usize, seed: u64) -> Vec<u16> {
        let mut v = vec![0u16; n];
        let mut s = seed | 1;
        for x in v.iter_mut() {
            s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            *x = 0x2C00u16.wrapping_add((s as u16) & 0x03FF);
        }
        v
    }

    fn fill_f32(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as u32).wrapping_add(seed) % 17) as f32 * 0.125 - 1.0)
            .collect()
    }

    fn det_w(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
        let mut w = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let k = r
                    .wrapping_mul(1315423911)
                    .wrapping_add(c)
                    .wrapping_add(seed as usize);
                w[r * cols + c] = ((k % 23) as f32) * 0.17 - 1.9;
            }
        }
        w
    }

    fn pack_binary(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u16>) {
        let gpr = cols / GROUP as usize;
        let mut scales = vec![0u16; rows * gpr];
        let mut bits = vec![0u8; (rows * cols + 7) / 8];
        for r in 0..rows {
            for g in 0..gpr {
                let mut s = 0.0f64;
                let base = r * cols + g * GROUP as usize;
                for k in 0..GROUP as usize {
                    s += f64::from(w[base + k].abs());
                    let flat = base + k;
                    if w[flat] >= 0.0 {
                        bits[flat >> 3] |= 1 << (flat & 7);
                    }
                }
                scales[r * gpr + g] = f16::from_f32((s / f64::from(GROUP)) as f32).to_bits();
            }
        }
        (bits, scales)
    }

    fn cpu_binary(signs: &[u8], scales: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let gpr = cols / GROUP as usize;
        let mut y = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                let flat = r * cols + c;
                let sc = f16::from_bits(scales[r * gpr + c / GROUP as usize]).to_f32();
                let pos = ((signs[flat >> 3] >> (flat & 7)) & 1) != 0;
                acc += (if pos { sc } else { -sc }) * x[c];
            }
            y[r] = acc;
        }
        y
    }

    fn pack_q2f(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u16>) {
        let gpr = cols / GROUP as usize;
        let mut delta = vec![0u16; rows * gpr];
        let mut q = vec![0u8; rows * cols];
        for r in 0..rows {
            for g in 0..gpr {
                let base = r * cols + g * GROUP as usize;
                let mut amax = 0.0f32;
                for k in 0..GROUP as usize {
                    amax = amax.max(w[base + k].abs());
                }
                let d = if amax > 0.0 { amax / 1.5 } else { 1.0 };
                delta[r * gpr + g] = f16::from_f32(d).to_bits();
                let d = f16::from_bits(delta[r * gpr + g]).to_f32();
                for k in 0..GROUP as usize {
                    let qq = if d.abs() > 0.0 {
                        ((w[base + k] / d) + 1.5).round().clamp(0.0, 3.0) as u8
                    } else {
                        0
                    };
                    q[base + k] = qq;
                }
            }
        }
        let mut codes = vec![0u8; (rows * cols) / 4];
        for (i, &qq) in q.iter().enumerate() {
            codes[i / 4] |= (qq & 3) << (2 * (i % 4));
        }
        (codes, delta)
    }

    fn cpu_q2f(codes: &[u8], delta: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let gpr = cols / GROUP as usize;
        let mut y = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                let flat = r * cols + c;
                let qq = (codes[flat / 4] >> (2 * (flat % 4))) & 3;
                let d = f16::from_bits(delta[r * gpr + c / GROUP as usize]).to_f32();
                acc += ((qq as f32) - 1.5) * d * x[c];
            }
            y[r] = acc;
        }
        y
    }

    fn cpu_lowrank(
        signs: &[u8],
        scales: &[u16],
        u: &[u16],
        v: &[u16],
        x: &[f32],
        rows: usize,
        cols: usize,
        rank: usize,
    ) -> Vec<f32> {
        let mut y = cpu_binary(signs, scales, x, rows, cols);
        let mut proj = vec![0.0f32; rank];
        for c in 0..cols {
            let xv = x[c];
            let vb = c * rank;
            for k in 0..rank {
                proj[k] += f16::from_bits(v[vb + k]).to_f32() * xv;
            }
        }
        for r in 0..rows {
            let ub = r * rank;
            let mut acc = 0.0f32;
            for k in 0..rank {
                acc += f16::from_bits(u[ub + k]).to_f32() * proj[k];
            }
            y[r] += acc;
        }
        y
    }

    fn cpu_shared_k2(
        signs: &[u8],
        scales: &[u16],
        s0: &[u8],
        s1: &[u8],
        escale: &[u16],
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let gpr = cols / GROUP as usize;
        let mut y = cpu_binary(signs, scales, x, rows, cols);
        let k1 = rows * gpr;
        for r in 0..rows {
            for c in 0..cols {
                let flat = r * cols + c;
                let g = r * gpr + c / GROUP as usize;
                let xv = x[c];
                let a0 = f16::from_bits(escale[g]).to_f32();
                let a1 = f16::from_bits(escale[k1 + g]).to_f32();
                let p0 = ((s0[flat >> 3] >> (flat & 7)) & 1) != 0;
                let p1 = ((s1[flat >> 3] >> (flat & 7)) & 1) != 0;
                y[r] += (if p0 { a0 } else { -a0 }) * xv;
                y[r] += (if p1 { a1 } else { -a1 }) * xv;
            }
        }
        y
    }

    fn geo_grid(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows / 2), 1, 1), MTLSize::new(128, 1, 1))
    }

    fn r8_grid(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows / 8), 1, 1), MTLSize::new(256, 1, 1))
    }

    fn grid_for(kernel: &str, rows: u32) -> (MTLSize, MTLSize) {
        if kernel.contains("r8_tg256") {
            r8_grid(rows)
        } else {
            geo_grid(rows)
        }
    }

    fn pipe(device: &Device, lib: &metal::LibraryRef, name: &str) -> ComputePipelineState {
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|e| fail(format!("{name}: {e}")));
        device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| fail(format!("pipeline {name}: {e}")))
    }

    fn compile<'a>(device: &'a Device, src: &str, label: &str) -> metal::Library {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        eprintln!("hybrid_operator: compile {label}");
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(src, &opts)
            .unwrap_or_else(|e| fail(format!("{label} shader compile: {e}")));
        eprintln!("  {label} compiled in {:.3}s", t0.elapsed().as_secs_f64());
        lib
    }

    fn dispatch_named(
        enc: &metal::ComputeCommandEncoderRef,
        p: &ComputePipelineState,
        kernel: &str,
        rows: u32,
        bind: impl Fn(&metal::ComputeCommandEncoderRef),
    ) {
        enc.set_compute_pipeline_state(p);
        bind(enc);
        let (g, t) = grid_for(kernel, rows);
        enc.dispatch_thread_groups(g, t);
    }

    struct Organ {
        name: &'static str,
        rows: u32,
        cols: u32,
        binary: &'static str,
        q2f: &'static str,
        lr8: &'static str,
        lr8_noop: &'static str,
        lr32: &'static str,
        lr32_noop: &'static str,
        shk2: &'static str,
        shk2_noop: &'static str,
    }

    fn organs() -> [Organ; 3] {
        [
            Organ {
                name: "gate",
                rows: GATE_ROWS,
                cols: GATE_COLS,
                binary: "binary_g64_matvec_geo_c5120_tpr64_tg128",
                q2f: "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                lr8: "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128",
                lr8_noop: "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128_noop",
                lr32: "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128",
                lr32_noop: "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128_noop",
                shk2: "binary_shared_k2_fused_geo_c5120_tpr64_tg128",
                shk2_noop: "binary_shared_k2_fused_geo_c5120_tpr64_tg128_noop",
            },
            Organ {
                name: "up",
                rows: GATE_ROWS,
                cols: GATE_COLS,
                binary: "binary_g64_matvec_geo_c5120_tpr64_tg128",
                q2f: "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                lr8: "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128",
                lr8_noop: "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128_noop",
                lr32: "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128",
                lr32_noop: "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128_noop",
                shk2: "binary_shared_k2_fused_geo_c5120_tpr64_tg128",
                shk2_noop: "binary_shared_k2_fused_geo_c5120_tpr64_tg128_noop",
            },
            Organ {
                name: "down",
                rows: DOWN_ROWS,
                cols: DOWN_COLS,
                binary: "binary_g64_matvec_geo_c17408_tpr64_tg128",
                q2f: "q2f_g64_matvec_geo_c17408_tpr64_tg128",
                lr8: "binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128",
                lr8_noop: "binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128_noop",
                lr32: "binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128",
                lr32_noop: "binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128_noop",
                shk2: "binary_shared_k2_fused_geo_c17408_tpr64_tg128",
                shk2_noop: "binary_shared_k2_fused_geo_c17408_tpr64_tg128_noop",
            },
        ]
    }

    struct PackedBinary {
        signs: Buffer,
        scales: Buffer,
        sign_stride: usize,
        scale_stride: usize,
        rows: u32,
        cols: u32,
        payload_bytes: u64,
    }

    fn build_binary(
        device: &Device,
        org: &Organ,
        layers: usize,
        skip_unique: bool,
        seed: u64,
    ) -> PackedBinary {
        let rows = org.rows as usize;
        let cols = org.cols as usize;
        let gpr = cols / GROUP as usize;
        let slen = (rows * cols + 7) / 8;
        let sc_len = rows * gpr;
        let n_unique = if skip_unique { 1 } else { layers };
        let sign_stride = align256(slen);
        let scale_stride = align256(sc_len * 2);
        PackedBinary {
            signs: new_buf(device, &fill_u8(sign_stride * n_unique, seed)),
            scales: new_buf(
                device,
                as_u8_u16(&fill_u16(scale_stride / 2 * n_unique, seed ^ 0xC0FFEE)),
            ),
            sign_stride,
            scale_stride,
            rows: org.rows,
            cols: org.cols,
            payload_bytes: (slen + sc_len * 2) as u64 * layers as u64,
        }
    }

    struct PackedQ2f {
        codes: Buffer,
        delta: Buffer,
        code_stride: usize,
        delta_stride: usize,
        rows: u32,
        payload_bytes: u64,
    }

    fn build_q2f(
        device: &Device,
        org: &Organ,
        layers: usize,
        skip_unique: bool,
        seed: u64,
    ) -> PackedQ2f {
        let rows = org.rows as usize;
        let cols = org.cols as usize;
        let gpr = cols / GROUP as usize;
        let clen = (rows * cols) / 4;
        let dlen = rows * gpr;
        let n_unique = if skip_unique { 1 } else { layers };
        let code_stride = align256(clen);
        let delta_stride = align256(dlen * 2);
        PackedQ2f {
            codes: new_buf(device, &fill_u8(code_stride * n_unique, seed)),
            delta: new_buf(
                device,
                as_u8_u16(&fill_u16(delta_stride / 2 * n_unique, seed ^ 0x1111)),
            ),
            code_stride,
            delta_stride,
            rows: org.rows,
            payload_bytes: (clen + dlen * 2) as u64 * layers as u64,
        }
    }

    struct PackedLr {
        signs: Buffer,
        scales: Buffer,
        u: Buffer,
        v: Buffer,
        sign_stride: usize,
        scale_stride: usize,
        u_stride: usize,
        v_stride: usize,
        rows: u32,
        rank: usize,
        payload_bytes: u64,
    }

    fn build_lr(
        device: &Device,
        org: &Organ,
        rank: usize,
        layers: usize,
        skip_unique: bool,
        seed: u64,
    ) -> PackedLr {
        let rows = org.rows as usize;
        let cols = org.cols as usize;
        let gpr = cols / GROUP as usize;
        let slen = (rows * cols + 7) / 8;
        let sc_len = rows * gpr;
        let ulen = rows * rank;
        let vlen = cols * rank;
        let n_unique = if skip_unique { 1 } else { layers };
        let sign_stride = align256(slen);
        let scale_stride = align256(sc_len * 2);
        let u_stride = align256(ulen * 2);
        let v_stride = align256(vlen * 2);
        PackedLr {
            signs: new_buf(device, &fill_u8(sign_stride * n_unique, seed)),
            scales: new_buf(
                device,
                as_u8_u16(&fill_u16(scale_stride / 2 * n_unique, seed ^ 0xA5A5)),
            ),
            u: new_buf(
                device,
                as_u8_u16(&fill_u16(u_stride / 2 * n_unique, seed ^ 0x1111)),
            ),
            v: new_buf(
                device,
                as_u8_u16(&fill_u16(v_stride / 2 * n_unique, seed ^ 0x2222)),
            ),
            sign_stride,
            scale_stride,
            u_stride,
            v_stride,
            rows: org.rows,
            rank,
            payload_bytes: (slen + sc_len * 2 + ulen * 2 + vlen * 2) as u64 * layers as u64,
        }
    }

    struct PackedShared {
        signs: Buffer,
        scales: Buffer,
        s0: Buffer,
        s1: Buffer,
        escale: Buffer,
        sign_stride: usize,
        scale_stride: usize,
        escale_stride: usize,
        rows: u32,
        payload_bytes: u64,
    }

    fn build_shared(
        device: &Device,
        org: &Organ,
        layers: usize,
        skip_unique: bool,
        seed: u64,
    ) -> PackedShared {
        let rows = org.rows as usize;
        let cols = org.cols as usize;
        let gpr = cols / GROUP as usize;
        let slen = (rows * cols + 7) / 8;
        let sc_len = rows * gpr;
        let n_unique = if skip_unique { 1 } else { layers };
        let sign_stride = align256(slen);
        let scale_stride = align256(sc_len * 2);
        let escale_stride = align256(sc_len * 2 * 2); // K=2
        PackedShared {
            signs: new_buf(device, &fill_u8(sign_stride * n_unique, seed)),
            scales: new_buf(
                device,
                as_u8_u16(&fill_u16(scale_stride / 2 * n_unique, seed ^ 0x3333)),
            ),
            // extra bases shared across layers: one plane each
            s0: new_buf(device, &fill_u8(slen, seed ^ 0xB0)),
            s1: new_buf(device, &fill_u8(slen, seed ^ 0xB1)),
            escale: new_buf(
                device,
                as_u8_u16(&fill_u16(escale_stride / 2 * n_unique, seed ^ 0x4444)),
            ),
            sign_stride,
            scale_stride,
            escale_stride,
            rows: org.rows,
            payload_bytes: (slen + sc_len * 2) as u64 * layers as u64
                + slen as u64 * 2
                + (sc_len * 2 * 2) as u64 * layers as u64,
        }
    }

    fn run_cmd_loop<F>(
        queue: &CommandQueue,
        n: usize,
        mut encode: F,
    ) -> (Vec<u64>, Vec<u64>)
    where
        F: FnMut(&metal::ComputeCommandEncoderRef),
    {
        let mut gpu = Vec::new();
        let mut wall = Vec::new();
        for _ in 0..n {
            let t0 = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            encode(enc);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            wall.push(t0.elapsed().as_nanos() as u64);
            if let Some(ns) = gpu_ns(cmd) {
                gpu.push(ns);
            }
        }
        (gpu, wall)
    }

    fn bench_binary(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        eprintln!("graph binary_g64 layers={layers}");
        let packed: Vec<PackedBinary> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| build_binary(device, o, layers, skip_unique, 0xA000 + i as u64))
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let n_unique = if skip_unique { 1 } else { layers };
        let run = |n: usize| {
            run_cmd_loop(queue, n, |enc| {
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let p = pipes.get(org.binary).unwrap();
                        let b = &packed[oi];
                        let u = layer % n_unique;
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        dispatch_named(enc, p, org.binary, b.rows, |e| {
                            e.set_buffer(0, Some(&b.signs), (u * b.sign_stride) as u64);
                            e.set_buffer(1, Some(&b.scales), (u * b.scale_stride) as u64);
                            e.set_buffer(2, Some(x), 0);
                            e.set_buffer(3, Some(y), 0);
                            set_u32(e, 4, b.rows);
                        });
                    }
                }
            })
        };
        let _ = run(warmup);
        let (gpu, wall) = run(reps);
        eprintln!("  binary geo median_gpu_ns={:?}", median_u64(gpu.clone()));
        let payload: u64 = packed.iter().map(|p| p.payload_bytes).sum();
        json!({
            "id": "binary_g64",
            "kind": "baseline",
            "fused_native_operator": true,
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "mlp_payload_bytes": payload,
            "dense_w_materialized": 0,
        })
    }

    fn bench_q2f(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        eprintln!("graph q2f_g64 layers={layers}");
        let packed: Vec<PackedQ2f> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| build_q2f(device, o, layers, skip_unique, 0xB000 + i as u64))
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let n_unique = if skip_unique { 1 } else { layers };
        let run = |n: usize| {
            run_cmd_loop(queue, n, |enc| {
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let p = pipes.get(org.q2f).unwrap();
                        let b = &packed[oi];
                        let u = layer % n_unique;
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        dispatch_named(enc, p, org.q2f, b.rows, |e| {
                            e.set_buffer(0, Some(&b.codes), (u * b.code_stride) as u64);
                            e.set_buffer(1, Some(&b.delta), (u * b.delta_stride) as u64);
                            e.set_buffer(2, Some(x), 0);
                            e.set_buffer(3, Some(y), 0);
                            set_u32(e, 4, b.rows);
                        });
                    }
                }
            })
        };
        let _ = run(warmup);
        let (gpu, wall) = run(reps);
        eprintln!("  q2f geo median_gpu_ns={:?}", median_u64(gpu.clone()));
        let payload: u64 = packed.iter().map(|p| p.payload_bytes).sum();
        json!({
            "id": "q2f_g64",
            "kind": "baseline",
            "fused_native_operator": true,
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "mlp_payload_bytes": payload,
            "dense_w_materialized": 0,
        })
    }

    fn bench_lr(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        rank: usize,
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        let id = format!("binary_lowrank_r{rank}");
        eprintln!("graph {id} layers={layers}");
        let packed: Vec<PackedLr> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| build_lr(device, o, rank, layers, skip_unique, 0xC000 + i as u64 + rank as u64))
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let n_unique = if skip_unique { 1 } else { layers };
        let kernel_of = |org: &Organ, noop: bool| -> &str {
            match (rank, noop) {
                (8, false) => org.lr8,
                (8, true) => org.lr8_noop,
                (32, false) => org.lr32,
                (32, true) => org.lr32_noop,
                _ => fail("rank must be 8 or 32"),
            }
        };
        let run = |noop: bool, n: usize| {
            run_cmd_loop(queue, n, |enc| {
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let name = kernel_of(org, noop);
                        let p = pipes.get(name).unwrap_or_else(|| fail(format!("missing {name}")));
                        let b = &packed[oi];
                        let u = layer % n_unique;
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        dispatch_named(enc, p, name, b.rows, |e| {
                            e.set_buffer(0, Some(&b.signs), (u * b.sign_stride) as u64);
                            e.set_buffer(1, Some(&b.scales), (u * b.scale_stride) as u64);
                            e.set_buffer(2, Some(&b.u), (u * b.u_stride) as u64);
                            e.set_buffer(3, Some(&b.v), (u * b.v_stride) as u64);
                            e.set_buffer(4, Some(x), 0);
                            e.set_buffer(5, Some(y), 0);
                        });
                    }
                }
            })
        };
        let _ = run(false, warmup);
        let (gpu, wall) = run(false, reps);
        eprintln!("  {id} geo median_gpu_ns={:?}", median_u64(gpu.clone()));
        let _ = run(true, warmup.min(2));
        let (ngpu, nwall) = run(true, reps);
        eprintln!("  {id} noop median_gpu_ns={:?}", median_u64(ngpu.clone()));
        let overlap = ranges_overlap(&gpu, &ngpu);
        let payload: u64 = packed.iter().map(|p| p.payload_bytes).sum();
        json!({
            "id": id,
            "kind": "hybrid_lowrank",
            "correction": "lowrank",
            "rank": rank,
            "fused_native_operator": true,
            "one_dispatch_per_gemv": true,
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "noop": {
                "gpu_ns": spread(&ngpu),
                "wall_ns": spread(&nwall),
                "overlap_with_fused": overlap,
            },
            "mlp_payload_bytes": payload,
            "dense_w_materialized": 0,
        })
    }

    fn bench_shared(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        eprintln!("graph binary_shared_k2 layers={layers}");
        let packed: Vec<PackedShared> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| build_shared(device, o, layers, skip_unique, 0xD000 + i as u64))
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let n_unique = if skip_unique { 1 } else { layers };
        let run = |noop: bool, n: usize| {
            run_cmd_loop(queue, n, |enc| {
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let name = if noop { org.shk2_noop } else { org.shk2 };
                        let p = pipes.get(name).unwrap();
                        let b = &packed[oi];
                        let u = layer % n_unique;
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        dispatch_named(enc, p, name, b.rows, |e| {
                            e.set_buffer(0, Some(&b.signs), (u * b.sign_stride) as u64);
                            e.set_buffer(1, Some(&b.scales), (u * b.scale_stride) as u64);
                            e.set_buffer(2, Some(&b.s0), 0);
                            e.set_buffer(3, Some(&b.s1), 0);
                            e.set_buffer(4, Some(&b.escale), (u * b.escale_stride) as u64);
                            e.set_buffer(5, Some(x), 0);
                            e.set_buffer(6, Some(y), 0);
                        });
                    }
                }
            })
        };
        let _ = run(false, warmup);
        let (gpu, wall) = run(false, reps);
        eprintln!(
            "  binary_shared_k2 geo median_gpu_ns={:?}",
            median_u64(gpu.clone())
        );
        let _ = run(true, warmup.min(2));
        let (ngpu, nwall) = run(true, reps);
        eprintln!(
            "  binary_shared_k2 noop median_gpu_ns={:?}",
            median_u64(ngpu.clone())
        );
        let overlap = ranges_overlap(&gpu, &ngpu);
        let payload: u64 = packed.iter().map(|p| p.payload_bytes).sum();
        json!({
            "id": "binary_shared_k2",
            "kind": "hybrid_shared_basis",
            "correction": "shared_basis",
            "k": 2,
            "fused_native_operator": true,
            "one_dispatch_per_gemv": true,
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "noop": {
                "gpu_ns": spread(&ngpu),
                "wall_ns": spread(&nwall),
                "overlap_with_fused": overlap,
            },
            "mlp_payload_bytes": payload,
            "dense_w_materialized": 0,
        })
    }

    fn run_parity(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
    ) -> Value {
        let mut rows_out = Vec::new();
        let rows = GATE_ROWS as usize;
        let cols = GATE_COLS as usize;
        let w = det_w(16, cols, 41);
        let mut w_full = vec![0.0f32; rows * cols];
        w_full[..16 * cols].copy_from_slice(&w);
        let x = fill_f32(cols, 7);
        let (bc, bs) = pack_binary(&w_full, rows, cols);
        let mut u8v = vec![0u16; rows * 8];
        let mut v8 = vec![0u16; cols * 8];
        for i in 0..16 * 8 {
            u8v[i] = f16::from_f32(((i % 11) as f32) * 0.01 - 0.05).to_bits();
        }
        for i in 0..cols * 8 {
            v8[i] = f16::from_f32(((i % 13) as f32) * 0.008 - 0.04).to_bits();
        }
        let cpu_lr = cpu_lowrank(&bc, &bs, &u8v, &v8, &x, 16, cols, 8);
        let (s0, _) = pack_binary(&det_w(rows, cols, 7), rows, cols);
        let (s1, _) = pack_binary(&det_w(rows, cols, 9), rows, cols);
        let mut esc = vec![0u16; rows * 80 * 2];
        for i in 0..esc.len() {
            esc[i] = f16::from_f32(0.02 + ((i % 5) as f32) * 0.001).to_bits();
        }
        let cpu_sh = cpu_shared_k2(&bc, &bs, &s0, &s1, &esc, &x, 16, cols);
        let (qc, qd) = pack_q2f(&w_full, rows, cols);
        let cpu_q = cpu_q2f(&qc, &qd, &x, 16, cols);
        let cpu_b = cpu_binary(&bc, &bs, &x, 16, cols);

        let bc_buf = new_buf(device, &bc);
        let bs_buf = new_buf(device, as_u8_u16(&bs));
        let u_buf = new_buf(device, as_u8_u16(&u8v));
        let v_buf = new_buf(device, as_u8_u16(&v8));
        let s0_buf = new_buf(device, &s0);
        let s1_buf = new_buf(device, &s1);
        let es_buf = new_buf(device, as_u8_u16(&esc));
        let qc_buf = new_buf(device, &qc);
        let qd_buf = new_buf(device, as_u8_u16(&qd));
        let x_buf = new_buf(device, as_u8_f32(&x));
        let y_buf = new_empty(device, rows * 4);

        let cases: [(&str, &str, Box<dyn Fn(&metal::ComputeCommandEncoderRef)>, &[f32]); 4] = [
            (
                "binary_geo_c5120",
                "binary_g64_matvec_geo_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&bc_buf), 0);
                    enc.set_buffer(1, Some(&bs_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&y_buf), 0);
                    set_u32(enc, 4, rows as u32);
                }),
                &cpu_b,
            ),
            (
                "q2f_geo_c5120",
                "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&qc_buf), 0);
                    enc.set_buffer(1, Some(&qd_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&y_buf), 0);
                    set_u32(enc, 4, rows as u32);
                }),
                &cpu_q,
            ),
            (
                "lowrank_r8_fused_c5120",
                "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&bc_buf), 0);
                    enc.set_buffer(1, Some(&bs_buf), 0);
                    enc.set_buffer(2, Some(&u_buf), 0);
                    enc.set_buffer(3, Some(&v_buf), 0);
                    enc.set_buffer(4, Some(&x_buf), 0);
                    enc.set_buffer(5, Some(&y_buf), 0);
                }),
                &cpu_lr,
            ),
            (
                "shared_k2_fused_c5120",
                "binary_shared_k2_fused_geo_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&bc_buf), 0);
                    enc.set_buffer(1, Some(&bs_buf), 0);
                    enc.set_buffer(2, Some(&s0_buf), 0);
                    enc.set_buffer(3, Some(&s1_buf), 0);
                    enc.set_buffer(4, Some(&es_buf), 0);
                    enc.set_buffer(5, Some(&x_buf), 0);
                    enc.set_buffer(6, Some(&y_buf), 0);
                }),
                &cpu_sh,
            ),
        ];

        for (id, kernel, bind, cpu) in cases {
            let p = pipes.get(kernel).unwrap();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p);
            bind(enc);
            let (g, t) = grid_for(kernel, rows as u32);
            enc.dispatch_thread_groups(g, t);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu = read_f32(&y_buf, 16);
            let diff = max_abs(&gpu, cpu);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!(
                "parity {id}: max_abs={diff:.4e} {}",
                if ok { "PASS" } else { "DIFF" }
            );
            if !ok {
                fail(format!("parity {id} max_abs={diff}"));
            }
            rows_out.push(json!({
                "id": id,
                "kernel": kernel,
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": true,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
                "n_rows_compared": 16,
            }));
        }

        // No-op must diverge from the fused body (loads only, no FMA).
        let p = pipes
            .get("binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128_noop")
            .unwrap();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(p);
        enc.set_buffer(0, Some(&bc_buf), 0);
        enc.set_buffer(1, Some(&bs_buf), 0);
        enc.set_buffer(2, Some(&u_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&x_buf), 0);
        enc.set_buffer(5, Some(&y_buf), 0);
        let (g, t) = r8_grid(rows as u32);
        enc.dispatch_thread_groups(g, t);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let gpu_noop = read_f32(&y_buf, 16);
        let noop_diff = max_abs(&gpu_noop, &cpu_lr);
        let noop_diverges = noop_diff > PASS_TOL;
        eprintln!("parity noop_diverges={noop_diverges} max_abs={noop_diff:.4e}");
        rows_out.push(json!({
            "id": "lowrank_r8_noop_diverges",
            "ok": noop_diverges,
            "noop_diverges": noop_diverges,
            "max_abs_vs_fused_cpu": noop_diff,
            "dense_w_materialized": 0,
        }));
        if !noop_diverges {
            fail("noop did not diverge from fused low-rank body");
        }
        json!(rows_out)
    }

    pub fn run(args: Args) {
        let t_all = Instant::now();
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal GPU"));
        let queue = device.new_command_queue();
        let lib_h = compile(&device, HYBRID_SHADER, "hybrid");
        let lib_b = compile(&device, BASE_SHADER, "bytes_frontier");

        let hybrid_names = [
            "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128",
            "binary_lowrank_r8_fused_xproj_c5120_tpr64_tg128_noop",
            "binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128",
            "binary_lowrank_r8_fused_xproj_c17408_tpr64_tg128_noop",
            "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128",
            "binary_lowrank_r32_fused_xproj_c5120_tpr64_tg128_noop",
            "binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128",
            "binary_lowrank_r32_fused_xproj_c17408_tpr64_tg128_noop",
            "binary_shared_k2_fused_geo_c5120_tpr64_tg128",
            "binary_shared_k2_fused_geo_c5120_tpr64_tg128_noop",
            "binary_shared_k2_fused_geo_c17408_tpr64_tg128",
            "binary_shared_k2_fused_geo_c17408_tpr64_tg128_noop",
        ];
        let base_names = [
            "binary_g64_matvec_geo_c5120_tpr64_tg128",
            "binary_g64_matvec_geo_c17408_tpr64_tg128",
            "q2f_g64_matvec_geo_c5120_tpr64_tg128",
            "q2f_g64_matvec_geo_c17408_tpr64_tg128",
        ];
        let mut pipes = std::collections::HashMap::new();
        let mut occupancy = Vec::new();
        for n in hybrid_names {
            let p = pipe(&device, &lib_h, n);
            occupancy.push(json!({
                "kernel": n,
                "file": "hybrid_operator.metal",
                "max_total_threads_per_threadgroup": p.max_total_threads_per_threadgroup(),
                "thread_execution_width": p.thread_execution_width(),
                "static_threadgroup_memory_length": p.static_threadgroup_memory_length(),
            }));
            pipes.insert(n, p);
        }
        for n in base_names {
            let p = pipe(&device, &lib_b, n);
            occupancy.push(json!({
                "kernel": n,
                "file": "bytes_frontier.metal",
                "max_total_threads_per_threadgroup": p.max_total_threads_per_threadgroup(),
                "thread_execution_width": p.thread_execution_width(),
                "static_threadgroup_memory_length": p.static_threadgroup_memory_length(),
            }));
            pipes.insert(n, p);
        }

        let parity = run_parity(&device, &queue, &pipes);
        let organs = organs();
        let mut graphs = Vec::new();
        graphs.push(bench_binary(
            &device,
            &queue,
            &pipes,
            &organs,
            args.warmup,
            args.reps,
            args.layers,
            args.skip_unique,
        ));
        graphs.push(bench_q2f(
            &device,
            &queue,
            &pipes,
            &organs,
            args.warmup,
            args.reps,
            args.layers,
            args.skip_unique,
        ));
        graphs.push(bench_lr(
            &device,
            &queue,
            &pipes,
            &organs,
            8,
            args.warmup,
            args.reps,
            args.layers,
            args.skip_unique,
        ));
        graphs.push(bench_lr(
            &device,
            &queue,
            &pipes,
            &organs,
            32,
            args.warmup,
            args.reps,
            args.layers,
            args.skip_unique,
        ));
        graphs.push(bench_shared(
            &device,
            &queue,
            &pipes,
            &organs,
            args.warmup,
            args.reps,
            args.layers,
            args.skip_unique,
        ));

        let doc = json!({
            "schema": "hawking.headless.hybrid_operator.raw.v1",
            "git_head": git_head(),
            "device": device.name().to_string(),
            "reps": args.reps,
            "warmup": args.warmup,
            "layers": args.layers,
            "skip_unique": args.skip_unique,
            "dense_w_materialized": 0,
            "n021_complete_gpu_ns_median": N021_COMPLETE_GPU_NS,
            "n032_q2f_mlp_gpu_ns_median": N032_Q2F_MLP_GPU_NS,
            "mlp_elements": MLP_ELEMENTS,
            "parent_params": PARENT_PARAMS,
            "q4_attn_f32_bytes": Q4_ATTN_F32_BYTES,
            "occupancy": occupancy,
            "parity": parity,
            "graphs": graphs,
            "elapsed_s": t_all.elapsed().as_secs_f64(),
        });
        let text = serde_json::to_string_pretty(&doc).expect("json");
        if let Some(path) = &args.out {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(path, &text).unwrap_or_else(|e| fail(e));
            eprintln!("wrote {}", path.display());
        }
        println!("{text}");
    }
}
