//! N032 — native kernels that move fewer ACTIVE bytes/token than q2f g64 (2.25 bpw).
//!
//! Three representations, each with a specialized Metal kernel (no dense W):
//!   ternary 5-in-8 + g64 scale (1.85 bpw stored = active; zeros are 0-FMA)
//!   K=2 shared binary bases (signs once per organ, per-layer scales)
//!   binary plane + 2% CSR residual, fused
//!
//! The speed experiment is a 64-layer unique-weight MLP token graph
//! (gate + up + down = 192 GEMVs) so DRAM is not a one-tensor cache hit.
//! COMPLETE_TOKEN_NS is that graph plus the N021 non-MLP residual.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example bytes_frontier
//! ./tools/gpu_lane_lock.sh n032-bytes \
//!   workspace/ops/build/rust/release-fast/examples/bytes_frontier \
//!   --reps 7 --out receipts/headless/_BYTES_FRONTIER_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

fn usage() -> &'static str {
    "usage: bytes_frontier [--reps N] [--warmup N] [--layers N] [--out FILE] [--skip-unique]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("bytes_frontier: {message}");
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
    fail("bytes_frontier is Metal-only");
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

    const SHADER: &str = include_str!("../shaders/bytes_frontier.metal");
    const PASS_TOL: f32 = 2e-2;
    const GATE_ROWS: u32 = 17408;
    const GATE_COLS: u32 = 5120;
    const DOWN_ROWS: u32 = 5120;
    const DOWN_COLS: u32 = 17408;
    const GROUP: u32 = 64;
    const K_SHARED: u32 = 2;
    const CSR_PCT: u32 = 2;
    const N021_COMPLETE_GPU_NS: u64 = 27_547_874;
    const MLP_ELEMENTS: u64 = 17_112_760_320;
    const PARENT_PARAMS: u64 = 26_895_998_464;
    const Q4_ATTN_F32_BYTES: u64 = 5_206_533_080;
    const ROOF_TOK_S: f64 = 729.7;

    fn align256(n: usize) -> usize {
        (n + 255) & !255
    }

    fn as_u8_u16(v: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
    }
    fn as_u8_u32(v: &[u32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
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
                let k = r.wrapping_mul(1315423911).wrapping_add(c).wrapping_add(seed as usize);
                w[r * cols + c] = ((k % 23) as f32) * 0.17 - 1.9;
            }
        }
        w
    }

    fn trit_of(w: f32, scale: f32) -> u8 {
        if w > scale * 0.5 {
            2
        } else if w < -scale * 0.5 {
            0
        } else {
            1
        }
    }

    fn pack_ternary(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u16>) {
        let gpr = cols / GROUP as usize;
        let mut scales = vec![0u16; rows * gpr];
        for r in 0..rows {
            for g in 0..gpr {
                let mut s = 0.0f64;
                let base = r * cols + g * GROUP as usize;
                for k in 0..GROUP as usize {
                    s += f64::from(w[base + k].abs());
                }
                let mean = (s / f64::from(GROUP)) as f32;
                scales[r * gpr + g] = f16::from_f32(mean).to_bits();
            }
        }
        let bpr = (cols + 4) / 5;
        let mut codes = vec![0u8; rows * bpr];
        for r in 0..rows {
            let mut col = 0usize;
            let mut bi = 0usize;
            while col < cols {
                let mut v = 0u32;
                let mut p = 1u32;
                for i in 0..5 {
                    let t = if col + i < cols {
                        let sc = f16::from_bits(scales[r * gpr + (col + i) / GROUP as usize]).to_f32();
                        trit_of(w[r * cols + col + i], sc)
                    } else {
                        1
                    };
                    v += u32::from(t) * p;
                    p *= 3;
                }
                codes[r * bpr + bi] = v as u8;
                bi += 1;
                col += 5;
            }
        }
        (codes, scales)
    }

    fn cpu_ternary(codes: &[u8], scales: &[u16], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let gpr = cols / GROUP as usize;
        let bpr = (cols + 4) / 5;
        let mut y = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let mut v = u32::from(codes[r * bpr + b]);
                let col0 = b * 5;
                for i in 0..5 {
                    let col = col0 + i;
                    if col >= cols {
                        break;
                    }
                    let t = v % 3;
                    v /= 3;
                    let w = (t as i32 - 1) as f32;
                    let sc = f16::from_bits(scales[r * gpr + col / GROUP as usize]).to_f32();
                    acc += (w * sc) * x[col];
                }
            }
            y[r] = acc;
        }
        y
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

    struct Csr {
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        corr: Vec<u16>,
    }

    fn pack_csr(rows: usize, cols: usize, seed: u32) -> Csr {
        let nnz_row = ((cols * CSR_PCT as usize) / 100).max(1);
        let mut row_ptr = vec![0u32; rows + 1];
        let mut col_idx = Vec::with_capacity(rows * nnz_row);
        let mut corr = Vec::with_capacity(rows * nnz_row);
        let mut s = u64::from(seed) | 1;
        for r in 0..rows {
            row_ptr[r] = col_idx.len() as u32;
            for j in 0..nnz_row {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                col_idx.push((s as usize % cols) as u32);
                corr.push(f16::from_f32((((s >> 33) as i32 % 17) as f32) * 0.01).to_bits());
                let _ = j;
            }
        }
        row_ptr[rows] = col_idx.len() as u32;
        Csr {
            row_ptr,
            col_idx,
            corr,
        }
    }

    fn cpu_residual(
        signs: &[u8],
        scales: &[u16],
        csr: &Csr,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut y = cpu_binary(signs, scales, x, rows, cols);
        for r in 0..rows {
            let begin = csr.row_ptr[r] as usize;
            let end = csr.row_ptr[r + 1] as usize;
            for j in begin..end {
                y[r] += f16::from_bits(csr.corr[j]).to_f32() * x[csr.col_idx[j] as usize];
            }
        }
        y
    }

    fn cpu_shared(
        s0: &[u8],
        s1: &[u8],
        scales: &[u16],
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let gpr = cols / GROUP as usize;
        let y0 = cpu_binary(s0, &scales[..rows * gpr], x, rows, cols);
        let y1 = cpu_binary(s1, &scales[rows * gpr..], x, rows, cols);
        y0.iter().zip(y1).map(|(a, b)| a + b).collect()
    }

    fn geo_grid(rows: u32) -> (MTLSize, MTLSize) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        (MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1))
    }

    fn serial_grid(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows), 1, 1), MTLSize::new(256, 1, 1))
    }

    fn pipe(device: &Device, lib: &metal::LibraryRef, name: &str) -> ComputePipelineState {
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|e| fail(format!("{name}: {e}")));
        device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| fail(format!("pipeline {name}: {e}")))
    }

    fn dispatch_geo(
        enc: &metal::ComputeCommandEncoderRef,
        p: &ComputePipelineState,
        rows: u32,
        bind: impl Fn(&metal::ComputeCommandEncoderRef),
    ) {
        enc.set_compute_pipeline_state(p);
        bind(enc);
        let (g, t) = geo_grid(rows);
        enc.dispatch_thread_groups(g, t);
    }

    fn dispatch_serial(
        enc: &metal::ComputeCommandEncoderRef,
        p: &ComputePipelineState,
        rows: u32,
        bind: impl Fn(&metal::ComputeCommandEncoderRef),
    ) {
        enc.set_compute_pipeline_state(p);
        bind(enc);
        let (g, t) = serial_grid(rows);
        enc.dispatch_threads(g, t);
    }

    pub fn run(args: Args) {
        let t_all = Instant::now();
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal GPU"));
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        eprintln!("bytes_frontier: compile shader");
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .unwrap_or_else(|e| fail(format!("shader compile: {e}")));
        let compile_s = t0.elapsed().as_secs_f64();
        eprintln!("  compiled in {compile_s:.3}s");

        let names = [
            "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
            "ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128",
            "ternary_5in8_g64_matvec_serial_c5120",
            "ternary_5in8_g64_matvec_serial_c17408",
            "binary_g64_matvec_geo_c5120_tpr64_tg128",
            "binary_g64_matvec_geo_c17408_tpr64_tg128",
            "binary_g64_matvec_serial_c5120",
            "binary_g64_matvec_serial_c17408",
            "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
            "shared_binary_k2_group_dots_c17408_g64_tpr64_tg128",
            "shared_binary_k2_scale_contract_gpr80",
            "shared_binary_k2_scale_contract_gpr272",
            "shared_binary_k2_reload_every_layer_c5120",
            "binary_sparse_fused_geo_c5120_tpr64_tg128",
            "binary_sparse_fused_geo_c17408_tpr64_tg128",
            "binary_sparse_fused_serial_c5120",
            "binary_sparse_noop_drop_csr_c5120",
            "q2f_g64_matvec_geo_c5120_tpr64_tg128",
            "q2f_g64_matvec_geo_c17408_tpr64_tg128",
            "q2f_g64_matvec_serial_c5120",
        ];
        let mut pipes = std::collections::HashMap::new();
        let mut occupancy = Vec::new();
        for n in names {
            let p = pipe(&device, &lib, n);
            occupancy.push(json!({
                "kernel": n,
                "max_total_threads_per_threadgroup": p.max_total_threads_per_threadgroup(),
                "thread_execution_width": p.thread_execution_width(),
            }));
            pipes.insert(n, p);
        }

        let parity = run_parity(&device, &queue, &pipes);
        let graphs = run_graphs(&device, &queue, &pipes, args.warmup, args.reps, args.layers, args.skip_unique);

        let doc = json!({
            "schema": "hawking.headless.bytes_frontier.raw.v1",
            "git_head": git_head(),
            "device": device.name().to_string(),
            "compile_s": compile_s,
            "reps": args.reps,
            "warmup": args.warmup,
            "layers": args.layers,
            "skip_unique": args.skip_unique,
            "dense_w_materialized": 0,
            "n021_complete_gpu_ns_median": N021_COMPLETE_GPU_NS,
            "mlp_elements": MLP_ELEMENTS,
            "parent_params": PARENT_PARAMS,
            "q4_attn_f32_bytes": Q4_ATTN_F32_BYTES,
            "roof_tok_s": ROOF_TOK_S,
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
        } else {
            println!("{text}");
        }
    }

    fn run_parity(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
    ) -> Value {
        let mut rows_out = Vec::new();
        // Gate-shaped parity, 16 rows of 5120.
        let pr = 16usize;
        let w = det_w(pr, GATE_COLS as usize, 41);
        let x = fill_f32(GATE_COLS as usize, 7);
        let (tc, ts) = pack_ternary(&w, pr, GATE_COLS as usize);
        let cpu_t = cpu_ternary(&tc, &ts, &x, pr, GATE_COLS as usize);
        let (bc, bs) = pack_binary(&w, pr, GATE_COLS as usize);
        let cpu_b = cpu_binary(&bc, &bs, &x, pr, GATE_COLS as usize);
        let (qc, qd) = pack_q2f(&w, pr, GATE_COLS as usize);
        let cpu_q = cpu_q2f(&qc, &qd, &x, pr, GATE_COLS as usize);
        let w2 = det_w(pr, GATE_COLS as usize, 97);
        let (b1, s1) = pack_binary(&w2, pr, GATE_COLS as usize);
        let mut sh_scales = bs.clone();
        sh_scales.extend_from_slice(&s1);
        let cpu_s = cpu_shared(&bc, &b1, &sh_scales, &x, pr, GATE_COLS as usize);
        let csr = pack_csr(pr, GATE_COLS as usize, 99);
        let cpu_r = cpu_residual(&bc, &bs, &csr, &x, pr, GATE_COLS as usize);

        let tc_buf = new_buf(device, &tc);
        let ts_buf = new_buf(device, as_u8_u16(&ts));
        let bc_buf = new_buf(device, &bc);
        let bs_buf = new_buf(device, as_u8_u16(&bs));
        let b1_buf = new_buf(device, &b1);
        let sh_buf = new_buf(device, as_u8_u16(&sh_scales));
        let qc_buf = new_buf(device, &qc);
        let qd_buf = new_buf(device, as_u8_u16(&qd));
        let x_buf = new_buf(device, as_u8_f32(&x));
        let rp_buf = new_buf(device, as_u8_u32(&csr.row_ptr));
        let ci_buf = new_buf(device, as_u8_u32(&csr.col_idx));
        let cr_buf = new_buf(device, as_u8_u16(&csr.corr));
        let y_buf = new_empty(device, pr * 4);

        let cases: &[(&str, &str, Box<dyn Fn(&metal::ComputeCommandEncoderRef)>, &[f32], bool)] = &[
            (
                "ternary_geo_c5120",
                "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&tc_buf), 0);
                    enc.set_buffer(1, Some(&ts_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&y_buf), 0);
                    set_u32(enc, 4, pr as u32);
                }),
                &cpu_t,
                true,
            ),
            (
                "binary_geo_c5120",
                "binary_g64_matvec_geo_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&bc_buf), 0);
                    enc.set_buffer(1, Some(&bs_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&y_buf), 0);
                    set_u32(enc, 4, pr as u32);
                }),
                &cpu_b,
                true,
            ),
            (
                "q2f_geo_c5120",
                "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&qc_buf), 0);
                    enc.set_buffer(1, Some(&qd_buf), 0);
                    enc.set_buffer(2, Some(&x_buf), 0);
                    enc.set_buffer(3, Some(&y_buf), 0);
                    set_u32(enc, 4, pr as u32);
                }),
                &cpu_q,
                true,
            ),
            (
                "residual_fused_c5120",
                "binary_sparse_fused_geo_c5120_tpr64_tg128",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&bc_buf), 0);
                    enc.set_buffer(1, Some(&bs_buf), 0);
                    enc.set_buffer(2, Some(&rp_buf), 0);
                    enc.set_buffer(3, Some(&ci_buf), 0);
                    enc.set_buffer(4, Some(&cr_buf), 0);
                    enc.set_buffer(5, Some(&x_buf), 0);
                    enc.set_buffer(6, Some(&y_buf), 0);
                    set_u32(enc, 7, pr as u32);
                }),
                &cpu_r,
                true,
            ),
            (
                "residual_noop_drop_csr",
                "binary_sparse_noop_drop_csr_c5120",
                Box::new(|enc| {
                    enc.set_buffer(0, Some(&bc_buf), 0);
                    enc.set_buffer(1, Some(&bs_buf), 0);
                    enc.set_buffer(2, Some(&rp_buf), 0);
                    enc.set_buffer(3, Some(&ci_buf), 0);
                    enc.set_buffer(4, Some(&cr_buf), 0);
                    enc.set_buffer(5, Some(&x_buf), 0);
                    enc.set_buffer(6, Some(&y_buf), 0);
                    set_u32(enc, 7, pr as u32);
                }),
                &cpu_r,
                false,
            ),
        ];

        for (id, kernel, bind, cpu, must_match) in cases {
            let p = pipes.get(kernel).unwrap();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p);
            bind(enc);
            let (g, t) = geo_grid(pr as u32);
            enc.dispatch_thread_groups(g, t);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu = read_f32(&y_buf, pr);
            let diff = max_abs(&gpu, cpu);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!(
                "parity {id}: max_abs={diff:.4e} {} (must_match={must_match})",
                if ok { "PASS" } else { "DIFF" }
            );
            if *must_match && !ok {
                fail(format!("parity {id} max_abs={diff}"));
            }
            if !*must_match && ok {
                fail(format!("noop {id} matched the residual oracle; bad control is vacuous"));
            }
            rows_out.push(json!({
                "id": id,
                "kernel": kernel,
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": must_match,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));
            let _ = sh_buf;
            let _ = b1_buf;
            let _ = cpu_s;
        }

        // Shared two-pass parity on 16 x 5120.
        {
            let gpr = GATE_COLS as usize / GROUP as usize;
            let dots_buf = new_empty(device, K_SHARED as usize * pr * gpr * 4);
            let p_dots = pipes
                .get("shared_binary_k2_group_dots_c5120_g64_tpr64_tg128")
                .unwrap();
            let p_sc = pipes.get("shared_binary_k2_scale_contract_gpr80").unwrap();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p_dots);
            enc.set_buffer(0, Some(&bc_buf), 0);
            enc.set_buffer(1, Some(&b1_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&dots_buf), 0);
            set_u32(enc, 4, pr as u32);
            let (g, t) = geo_grid(pr as u32);
            enc.dispatch_thread_groups(g, t);
            enc.set_compute_pipeline_state(p_sc);
            enc.set_buffer(0, Some(&sh_buf), 0);
            enc.set_buffer(1, Some(&dots_buf), 0);
            enc.set_buffer(2, Some(&y_buf), 0);
            set_u32(enc, 3, pr as u32);
            let (sg, st) = serial_grid(pr as u32);
            enc.dispatch_threads(sg, st);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu = read_f32(&y_buf, pr);
            let diff = max_abs(&gpu, &cpu_s);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!("parity shared_k2: max_abs={diff:.4e} {}", if ok { "PASS" } else { "FAIL" });
            if !ok {
                fail(format!("shared k2 parity max_abs={diff}"));
            }
            rows_out.push(json!({
                "id": "shared_k2_two_pass_c5120",
                "kernel": "shared_binary_k2_group_dots + scale_contract_gpr80",
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": true,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));
        }

        json!(rows_out)
    }

    struct Organ {
        name: &'static str,
        rows: u32,
        cols: u32,
        geo: &'static str,
        serial: &'static str,
        dots: &'static str,
        contract: &'static str,
        residual_geo: &'static str,
    }

    fn organs() -> [Organ; 3] {
        [
            Organ {
                name: "gate",
                rows: GATE_ROWS,
                cols: GATE_COLS,
                geo: "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
                serial: "ternary_5in8_g64_matvec_serial_c5120",
                dots: "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
                contract: "shared_binary_k2_scale_contract_gpr80",
                residual_geo: "binary_sparse_fused_geo_c5120_tpr64_tg128",
            },
            Organ {
                name: "up",
                rows: GATE_ROWS,
                cols: GATE_COLS,
                geo: "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
                serial: "ternary_5in8_g64_matvec_serial_c5120",
                dots: "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
                contract: "shared_binary_k2_scale_contract_gpr80",
                residual_geo: "binary_sparse_fused_geo_c5120_tpr64_tg128",
            },
            Organ {
                name: "down",
                rows: DOWN_ROWS,
                cols: DOWN_COLS,
                geo: "ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128",
                serial: "ternary_5in8_g64_matvec_serial_c17408",
                dots: "shared_binary_k2_group_dots_c17408_g64_tpr64_tg128",
                contract: "shared_binary_k2_scale_contract_gpr272",
                residual_geo: "binary_sparse_fused_geo_c17408_tpr64_tg128",
            },
        ]
    }

    fn ternary_bpr(cols: u32) -> usize {
        (cols as usize + 4) / 5
    }

    fn run_graphs(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        let organs = organs();
        let mut out = Vec::new();
        out.push(bench_packed_mlp(
            device, queue, pipes, &organs, "q2f", warmup, reps, layers, skip_unique,
        ));
        out.push(bench_packed_mlp(
            device, queue, pipes, &organs, "ternary", warmup, reps, layers, skip_unique,
        ));
        out.push(bench_packed_mlp(
            device, queue, pipes, &organs, "binary", warmup, reps, layers, skip_unique,
        ));
        out.push(bench_shared(
            device, queue, pipes, &organs, warmup, reps, layers,
        ));
        out.push(bench_residual(
            device, queue, pipes, &organs, warmup, reps, layers, skip_unique,
        ));
        json!(out)
    }

    struct PackedOrgan {
        codes: Buffer,
        scales: Buffer,
        code_stride: usize,
        scale_stride: usize,
        rows: u32,
        cols: u32,
        payload_bytes: u64,
    }

    fn build_packed(
        device: &Device,
        org: &Organ,
        kind: &str,
        layers: usize,
        skip_unique: bool,
        seed: u64,
    ) -> PackedOrgan {
        let gpr = (org.cols / GROUP) as usize;
        let (code_len, scale_len) = match kind {
            "q2f" => ((org.rows as usize * org.cols as usize) / 4, org.rows as usize * gpr),
            "ternary" => (org.rows as usize * ternary_bpr(org.cols), org.rows as usize * gpr),
            "binary" => (
                (org.rows as usize * org.cols as usize + 7) / 8,
                org.rows as usize * gpr,
            ),
            _ => fail("unknown pack kind"),
        };
        let n_unique = if skip_unique { 1 } else { layers };
        let code_stride = align256(code_len);
        let scale_stride = align256(scale_len * 2);
        let mut codes = fill_u8(code_stride * n_unique, seed);
        let scales = fill_u16(scale_stride / 2 * n_unique, seed ^ 0xC0FFEE);
        // Keep 5-in-8 bytes in range (max 242).
        if kind == "ternary" {
            for b in codes.iter_mut() {
                *b %= 243;
            }
        }
        PackedOrgan {
            codes: new_buf(device, &codes),
            scales: new_buf(device, as_u8_u16(&scales)),
            code_stride,
            scale_stride,
            rows: org.rows,
            cols: org.cols,
            payload_bytes: (code_len + scale_len * 2) as u64 * layers as u64,
        }
    }

    fn bind_matvec(
        enc: &metal::ComputeCommandEncoderRef,
        packed: &PackedOrgan,
        x: &Buffer,
        y: &Buffer,
        layer: usize,
        n_unique: usize,
    ) {
        let u = layer % n_unique;
        enc.set_buffer(0, Some(&packed.codes), (u * packed.code_stride) as u64);
        enc.set_buffer(1, Some(&packed.scales), (u * packed.scale_stride) as u64);
        enc.set_buffer(2, Some(x), 0);
        enc.set_buffer(3, Some(y), 0);
        set_u32(enc, 4, packed.rows);
    }

    fn bench_packed_mlp(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        kind: &str,
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        eprintln!("graph {kind} layers={layers} unique={}", !skip_unique);
        let packed: Vec<PackedOrgan> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| build_packed(device, o, kind, layers, skip_unique, 0xA000 + i as u64))
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let n_unique = if skip_unique { 1 } else { layers };
        let geo_names = match kind {
            "q2f" => [
                "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                "q2f_g64_matvec_geo_c17408_tpr64_tg128",
            ],
            "ternary" => [
                "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
                "ternary_5in8_g64_matvec_geo_c5120_tpr64_tg128",
                "ternary_5in8_g64_matvec_geo_c17408_tpr64_tg128",
            ],
            "binary" => [
                "binary_g64_matvec_geo_c5120_tpr64_tg128",
                "binary_g64_matvec_geo_c5120_tpr64_tg128",
                "binary_g64_matvec_geo_c17408_tpr64_tg128",
            ],
            _ => fail("kind"),
        };
        let serial_names = match kind {
            "q2f" => [
                "q2f_g64_matvec_serial_c5120",
                "q2f_g64_matvec_serial_c5120",
                "binary_g64_matvec_serial_c17408",
            ],
            "ternary" => [
                organs[0].serial,
                organs[1].serial,
                organs[2].serial,
            ],
            "binary" => [
                "binary_g64_matvec_serial_c5120",
                "binary_g64_matvec_serial_c5120",
                "binary_g64_matvec_serial_c17408",
            ],
            _ => fail("kind"),
        };
        // q2f serial down: we only specialized serial c5120 for q2f; use geo for down serial skip.
        let run = |serial: bool, n: usize| -> (Vec<u64>, Vec<u64>) {
            let mut gpu = Vec::new();
            let mut wall = Vec::new();
            for _ in 0..n {
                let t0 = Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let name = if serial { serial_names[oi] } else { geo_names[oi] };
                        if kind == "q2f" && serial && oi == 2 {
                            continue;
                        }
                        let p = pipes.get(name).unwrap();
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        if serial {
                            dispatch_serial(enc, p, packed[oi].rows, |e| {
                                bind_matvec(e, &packed[oi], x, y, layer, n_unique)
                            });
                        } else {
                            dispatch_geo(enc, p, packed[oi].rows, |e| {
                                bind_matvec(e, &packed[oi], x, y, layer, n_unique)
                            });
                        }
                    }
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                wall.push(t0.elapsed().as_nanos() as u64);
                if let Some(ns) = gpu_ns(cmd) {
                    gpu.push(ns);
                }
            }
            (gpu, wall)
        };
        let _ = run(false, warmup);
        let (gpu, wall) = run(false, reps);
        eprintln!("  {kind} geo median_gpu_ns={:?}", median_u64(gpu.clone()));
        let mut serial_json = Value::Null;
        if kind != "q2f" {
            let _ = run(true, warmup.min(2));
            let (sgpu, swall) = run(true, reps);
            eprintln!("  {kind} serial median_gpu_ns={:?}", median_u64(sgpu.clone()));
            serial_json = json!({
                "gpu_ns": spread(&sgpu),
                "wall_ns": spread(&swall),
                "overlap_with_geo": ranges_overlap(&gpu, &sgpu),
            });
        }
        let payload: u64 = packed.iter().map(|p| p.payload_bytes).sum();
        json!({
            "id": kind,
            "role": if kind == "q2f" { "baseline_2.25_bpw" } else { "candidate" },
            "layers": layers,
            "unique_weight_tensors": if skip_unique { 3 } else { 3 * layers },
            "dispatches": layers * if kind == "q2f" { 3 } else { 3 },
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "serial": serial_json,
            "mlp_payload_bytes": payload,
            "dense_w_materialized": 0,
            "kernels": geo_names,
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
    ) -> Value {
        eprintln!("graph shared_k2 layers={layers}");
        struct SharedOrg {
            s0: Buffer,
            s1: Buffer,
            scales: Buffer,
            scale_stride: usize,
            dots: Buffer,
            rows: u32,
            cols: u32,
            sign_bytes: u64,
            scale_bytes: u64,
        }
        let built: Vec<SharedOrg> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let sign_len = (o.rows as usize * o.cols as usize + 7) / 8;
                let gpr = (o.cols / GROUP) as usize;
                let scale_len = K_SHARED as usize * o.rows as usize * gpr;
                let scale_stride = align256(scale_len * 2);
                let s0 = fill_u8(sign_len, 0xB000 + i as u64);
                let s1 = fill_u8(sign_len, 0xB100 + i as u64);
                let scales = fill_u16((scale_stride / 2) * layers, 0xB200 + i as u64);
                let dots = new_empty(device, K_SHARED as usize * o.rows as usize * gpr * 4);
                SharedOrg {
                    s0: new_buf(device, &s0),
                    s1: new_buf(device, &s1),
                    scales: new_buf(device, as_u8_u16(&scales)),
                    scale_stride,
                    dots,
                    rows: o.rows,
                    cols: o.cols,
                    sign_bytes: (sign_len * 2) as u64,
                    scale_bytes: (scale_len * 2 * layers) as u64,
                }
            })
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);

        let run = |reload: bool, n: usize| -> (Vec<u64>, Vec<u64>) {
            let mut gpu = Vec::new();
            let mut wall = Vec::new();
            for _ in 0..n {
                let t0 = Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let b = &built[oi];
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        if reload && oi == 0 {
                            let p = pipes
                                .get("shared_binary_k2_reload_every_layer_c5120")
                                .unwrap();
                            dispatch_geo(enc, p, b.rows, |e| {
                                e.set_buffer(0, Some(&b.s0), 0);
                                e.set_buffer(1, Some(&b.s1), 0);
                                e.set_buffer(2, Some(&b.scales), (layer * b.scale_stride) as u64);
                                e.set_buffer(3, Some(x), 0);
                                e.set_buffer(4, Some(y), 0);
                                set_u32(e, 5, b.rows);
                            });
                            continue;
                        }
                        if reload && oi != 0 {
                            // down/up still two-pass when reload control is gate-only
                        }
                        let p_dots = pipes.get(org.dots).unwrap();
                        dispatch_geo(enc, p_dots, b.rows, |e| {
                            e.set_buffer(0, Some(&b.s0), 0);
                            e.set_buffer(1, Some(&b.s1), 0);
                            e.set_buffer(2, Some(x), 0);
                            e.set_buffer(3, Some(&b.dots), 0);
                            set_u32(e, 4, b.rows);
                        });
                        let p_sc = pipes.get(org.contract).unwrap();
                        dispatch_serial(enc, p_sc, b.rows, |e| {
                            e.set_buffer(0, Some(&b.scales), (layer * b.scale_stride) as u64);
                            e.set_buffer(1, Some(&b.dots), 0);
                            e.set_buffer(2, Some(y), 0);
                            set_u32(e, 3, b.rows);
                        });
                    }
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                wall.push(t0.elapsed().as_nanos() as u64);
                if let Some(ns) = gpu_ns(cmd) {
                    gpu.push(ns);
                }
            }
            (gpu, wall)
        };
        let _ = run(false, warmup);
        let (gpu, wall) = run(false, reps);
        eprintln!("  shared amortized median_gpu_ns={:?}", median_u64(gpu.clone()));
        let _ = run(true, warmup.min(2));
        let (r_gpu, r_wall) = run(true, reps);
        eprintln!(
            "  shared reload-every-layer (gate fused) median_gpu_ns={:?}",
            median_u64(r_gpu.clone())
        );
        let sign_bytes: u64 = built.iter().map(|b| b.sign_bytes).sum();
        let scale_bytes: u64 = built.iter().map(|b| b.scale_bytes).sum();
        json!({
            "id": "shared_binary_k2",
            "role": "candidate",
            "layers": layers,
            "dispatches": layers * 3 * 2,
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "reload_control": {
                "gpu_ns": spread(&r_gpu),
                "wall_ns": spread(&r_wall),
                "overlap_with_amortized": ranges_overlap(&gpu, &r_gpu),
                "note": "reload control fuses K=2 on gate only; up/down stay two-pass",
            },
            "basis_sign_bytes": sign_bytes,
            "per_layer_scale_bytes_total": scale_bytes,
            "mlp_payload_bytes_active_fused": sign_bytes + scale_bytes,
            "dense_w_materialized": 0,
        })
    }

    fn bench_residual(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        eprintln!("graph residual_2pct layers={layers}");
        struct ResOrg {
            signs: Buffer,
            scales: Buffer,
            row_ptr: Buffer,
            col_idx: Buffer,
            corr: Buffer,
            sign_stride: usize,
            scale_stride: usize,
            ptr_stride: usize,
            idx_stride: usize,
            corr_stride: usize,
            rows: u32,
            payload_bytes: u64,
        }
        let n_unique = if skip_unique { 1 } else { layers.min(8) };
        // CSR of 64 unique full organs is ~2GB extra; cap unique CSR tiles at 8
        // and still unique-ify signs per layer so DRAM is not one-tensor.
        let built: Vec<ResOrg> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let sign_len = (o.rows as usize * o.cols as usize + 7) / 8;
                let gpr = (o.cols / GROUP) as usize;
                let scale_len = o.rows as usize * gpr;
                let sign_stride = align256(sign_len);
                let scale_stride = align256(scale_len * 2);
                let signs = fill_u8(sign_stride * layers, 0xC000 + i as u64);
                let scales = fill_u16((scale_stride / 2) * layers, 0xC100 + i as u64);
                let csr = pack_csr(o.rows as usize, o.cols as usize, 200 + i as u32);
                let ptr_stride = align256(csr.row_ptr.len() * 4);
                let idx_stride = align256(csr.col_idx.len() * 4);
                let corr_stride = align256(csr.corr.len() * 2);
                let mut ptrs = vec![0u8; ptr_stride * n_unique];
                let mut idxs = vec![0u8; idx_stride * n_unique];
                let mut corrs = vec![0u8; corr_stride * n_unique];
                for u in 0..n_unique {
                    let csr_u = if u == 0 {
                        csr.row_ptr.clone()
                    } else {
                        pack_csr(o.rows as usize, o.cols as usize, 200 + i as u32 + u as u32).row_ptr
                    };
                    let c2 = pack_csr(o.rows as usize, o.cols as usize, 300 + i as u32 + u as u32);
                    ptrs[u * ptr_stride..u * ptr_stride + csr_u.len() * 4]
                        .copy_from_slice(as_u8_u32(&c2.row_ptr));
                    idxs[u * idx_stride..u * idx_stride + c2.col_idx.len() * 4]
                        .copy_from_slice(as_u8_u32(&c2.col_idx));
                    corrs[u * corr_stride..u * corr_stride + c2.corr.len() * 2]
                        .copy_from_slice(as_u8_u16(&c2.corr));
                }
                let payload = (sign_len + scale_len * 2) as u64 * layers as u64
                    + (csr.col_idx.len() * 4 + csr.corr.len() * 2 + csr.row_ptr.len() * 4) as u64
                        * layers as u64;
                ResOrg {
                    signs: new_buf(device, &signs),
                    scales: new_buf(device, as_u8_u16(&scales)),
                    row_ptr: new_buf(device, &ptrs),
                    col_idx: new_buf(device, &idxs),
                    corr: new_buf(device, &corrs),
                    sign_stride,
                    scale_stride,
                    ptr_stride,
                    idx_stride,
                    corr_stride,
                    rows: o.rows,
                    payload_bytes: payload,
                }
            })
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);

        let run = |noop: bool, n: usize| -> (Vec<u64>, Vec<u64>) {
            let mut gpu = Vec::new();
            let mut wall = Vec::new();
            for _ in 0..n {
                let t0 = Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let b = &built[oi];
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        let kernel = if noop && oi == 0 {
                            "binary_sparse_noop_drop_csr_c5120"
                        } else {
                            org.residual_geo
                        };
                        let p = pipes.get(kernel).unwrap();
                        let u = layer % n_unique;
                        dispatch_geo(enc, p, b.rows, |e| {
                            e.set_buffer(0, Some(&b.signs), (layer * b.sign_stride) as u64);
                            e.set_buffer(1, Some(&b.scales), (layer * b.scale_stride) as u64);
                            e.set_buffer(2, Some(&b.row_ptr), (u * b.ptr_stride) as u64);
                            e.set_buffer(3, Some(&b.col_idx), (u * b.idx_stride) as u64);
                            e.set_buffer(4, Some(&b.corr), (u * b.corr_stride) as u64);
                            e.set_buffer(5, Some(x), 0);
                            e.set_buffer(6, Some(y), 0);
                            set_u32(e, 7, b.rows);
                        });
                    }
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                wall.push(t0.elapsed().as_nanos() as u64);
                if let Some(ns) = gpu_ns(cmd) {
                    gpu.push(ns);
                }
            }
            (gpu, wall)
        };
        let _ = run(false, warmup);
        let (gpu, wall) = run(false, reps);
        eprintln!("  residual fused median_gpu_ns={:?}", median_u64(gpu.clone()));
        let _ = run(true, warmup.min(2));
        let (n_gpu, n_wall) = run(true, reps);
        eprintln!(
            "  residual noop-drop-csr (gate) median_gpu_ns={:?}",
            median_u64(n_gpu.clone())
        );
        let payload: u64 = built.iter().map(|b| b.payload_bytes).sum();
        json!({
            "id": "binary_residual_sparse_2pct",
            "role": "candidate",
            "layers": layers,
            "csr_unique_tiles": n_unique,
            "dispatches": layers * 3,
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "noop_drop_csr": {
                "gpu_ns": spread(&n_gpu),
                "wall_ns": spread(&n_wall),
                "overlap_with_fused": ranges_overlap(&gpu, &n_gpu),
            },
            "mlp_payload_bytes": payload,
            "dense_w_materialized": 0,
        })
    }
}
