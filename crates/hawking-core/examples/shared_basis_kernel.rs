//! N033 — competent native kernel for K=2 shared binary bases.
//!
//! Before: N032 two-pass (group-dots + scale-contract) = 384 dispatches,
//! per-group threadgroup barriers. After: one fused operator, specialized
//! K=2 / group-64, x (and signs, on c5120) staged in threadgroup memory,
//! simd reduction, 192 dispatches (or 3 if signs are reused across 64 layers).
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example shared_basis_kernel
//! ./tools/gpu_lane_lock.sh n033-sharedbasis \
//!   workspace/ops/build/rust/release-fast/examples/shared_basis_kernel \
//!   --reps 7 --out receipts/headless/_SHARED_BASIS_KERNEL_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

fn usage() -> &'static str {
    "usage: shared_basis_kernel [--reps N] [--warmup N] [--layers N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("shared_basis_kernel: {message}");
    process::exit(2);
}

struct Args {
    reps: usize,
    warmup: usize,
    layers: usize,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut reps = 7usize;
    let mut warmup = 2usize;
    let mut layers = 64usize;
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
    fail("shared_basis_kernel is Metal-only");
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

    const NEW_SHADER: &str = include_str!("../shaders/shared_basis_kernel.metal");
    const OLD_SHADER: &str = include_str!("../shaders/bytes_frontier.metal");
    const PASS_TOL: f32 = 2e-2;
    const GATE_ROWS: u32 = 17408;
    const GATE_COLS: u32 = 5120;
    const DOWN_ROWS: u32 = 5120;
    const DOWN_COLS: u32 = 17408;
    const GROUP: u32 = 64;
    const K_SHARED: u32 = 2;
    const N021_COMPLETE_GPU_NS: u64 = 27_547_874;
    const N032_Q2F_MLP_GPU_NS: u64 = 15_738_249;
    const N032_TWOPASS_MLP_GPU_NS: u64 = 99_816_541;
    const N032_TWOPASS_DISPATCHES: u64 = 384;
    const MLP_ELEMENTS: u64 = 17_112_760_320;
    const Q4_ATTN_F32_BYTES: u64 = 5_206_533_080;
    const ROOF_TOK_S: f64 = 729.7;

    const FUSED_C5120: &[&str] = &[
        "shared_binary_k2_fused_xsign_c5120_r8_tg256",
        "shared_binary_k2_fused_stream_c5120_tpr32_tg256",
        "shared_binary_k2_fused_stream_c5120_tpr64_tg128",
    ];
    const FUSED_C17408: &[&str] = &[
        "shared_binary_k2_fused_xtile_c17408_r8_tg256",
        "shared_binary_k2_fused_stream_c17408_tpr32_tg256",
        "shared_binary_k2_fused_stream_c17408_tpr64_tg128",
    ];

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
                let k = r.wrapping_mul(1315423911).wrapping_add(c).wrapping_add(seed as usize);
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

    fn cpu_shared(
        s0: &[u8],
        s1: &[u8],
        scales: &[u16],
        x: &[f32],
        rows: usize,
        cols: usize,
        scale_rows: usize,
    ) -> Vec<f32> {
        let gpr = cols / GROUP as usize;
        let y0 = cpu_binary(s0, &scales[..scale_rows * gpr], x, rows, cols);
        let y1 = cpu_binary(s1, &scales[scale_rows * gpr..], x, rows, cols);
        y0.iter().zip(y1).map(|(a, b)| a + b).collect()
    }

    fn grid_r8(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows / 8), 1, 1), MTLSize::new(256, 1, 1))
    }
    fn grid_tpr64(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows / 2), 1, 1), MTLSize::new(128, 1, 1))
    }
    fn grid_serial(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows), 1, 1), MTLSize::new(256, 1, 1))
    }

    fn grid_for(kernel: &str, rows: u32) -> (MTLSize, MTLSize) {
        if kernel.contains("serial") {
            grid_serial(rows)
        } else if kernel.contains("tpr64") || kernel.contains("group_dots") {
            grid_tpr64(rows)
        } else if kernel.contains("scale_contract") {
            grid_serial(rows)
        } else {
            grid_r8(rows)
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

    fn dispatch(
        enc: &metal::ComputeCommandEncoderRef,
        p: &ComputePipelineState,
        kernel: &str,
        rows: u32,
        bind: impl Fn(&metal::ComputeCommandEncoderRef),
    ) {
        enc.set_compute_pipeline_state(p);
        bind(enc);
        let (g, t) = grid_for(kernel, rows);
        if kernel.contains("serial") || kernel.contains("scale_contract") {
            enc.dispatch_threads(g, t);
        } else {
            enc.dispatch_thread_groups(g, t);
        }
    }

    fn compile<'a>(device: &'a Device, src: &str, label: &str) -> metal::Library {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        eprintln!("shared_basis_kernel: compile {label}");
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(src, &opts)
            .unwrap_or_else(|e| fail(format!("{label} shader compile: {e}")));
        eprintln!("  {label} compiled in {:.3}s", t0.elapsed().as_secs_f64());
        lib
    }

    pub fn run(args: Args) {
        let t_all = Instant::now();
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal GPU"));
        let queue = device.new_command_queue();
        let lib_new = compile(&device, NEW_SHADER, "fused");
        let lib_old = compile(&device, OLD_SHADER, "two-pass");

        let mut pipes = std::collections::HashMap::new();
        let mut occupancy = Vec::new();
        let new_names = [
            "shared_binary_k2_fused_xsign_c5120_r8_tg256",
            "shared_binary_k2_fused_xsign_c5120_r8_tg256_noop",
            "shared_binary_k2_fused_stream_c5120_tpr32_tg256",
            "shared_binary_k2_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k2_fused_serial_c5120",
            "shared_binary_k2_fused_xsign_layers64_c5120_r8_tg256",
            "shared_binary_k2_fused_xtile_c17408_r8_tg256",
            "shared_binary_k2_fused_stream_c17408_tpr32_tg256",
            "shared_binary_k2_fused_stream_c17408_tpr64_tg128",
            "shared_binary_k2_fused_serial_c17408",
            "shared_binary_k2_fused_xtile_layers64_c17408_r8_tg256",
            "shared_binary_k4_fused_stream_c5120_tpr32_tg256",
        ];
        for n in new_names {
            let p = pipe(&device, &lib_new, n);
            occupancy.push(json!({
                "kernel": n,
                "max_total_threads_per_threadgroup": p.max_total_threads_per_threadgroup(),
                "thread_execution_width": p.thread_execution_width(),
                "static_threadgroup_memory_length": p.static_threadgroup_memory_length(),
            }));
            pipes.insert(n, p);
        }
        let old_names = [
            "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128",
            "shared_binary_k2_group_dots_c17408_g64_tpr64_tg128",
            "shared_binary_k2_scale_contract_gpr80",
            "shared_binary_k2_scale_contract_gpr272",
        ];
        for n in old_names {
            let p = pipe(&device, &lib_old, n);
            occupancy.push(json!({
                "kernel": n,
                "max_total_threads_per_threadgroup": p.max_total_threads_per_threadgroup(),
                "thread_execution_width": p.thread_execution_width(),
                "static_threadgroup_memory_length": p.static_threadgroup_memory_length(),
            }));
            pipes.insert(n, p);
        }

        let parity = run_parity(&device, &queue, &pipes);
        let geo = run_geometry_search(&device, &queue, &pipes, args.warmup, args.reps);
        let winner_c5120 = geo["winner_c5120"]["kernel"].as_str().unwrap().to_string();
        let winner_c17408 = geo["winner_c17408"]["kernel"].as_str().unwrap().to_string();
        eprintln!("winners: c5120={winner_c5120}  c17408={winner_c17408}");
        let graphs = run_graphs(
            &device,
            &queue,
            &pipes,
            args.warmup,
            args.reps,
            args.layers,
            &winner_c5120,
            &winner_c17408,
        );

        let doc = json!({
            "schema": "hawking.headless.shared_basis_kernel.raw.v1",
            "git_head": git_head(),
            "device": device.name().to_string(),
            "reps": args.reps,
            "warmup": args.warmup,
            "layers": args.layers,
            "dense_w_materialized": 0,
            "n021_complete_gpu_ns_median": N021_COMPLETE_GPU_NS,
            "n032_q2f_mlp_gpu_ns_median": N032_Q2F_MLP_GPU_NS,
            "n032_twopass_mlp_gpu_ns_median": N032_TWOPASS_MLP_GPU_NS,
            "n032_twopass_dispatches": N032_TWOPASS_DISPATCHES,
            "mlp_elements": MLP_ELEMENTS,
            "q4_attn_f32_bytes": Q4_ATTN_F32_BYTES,
            "roof_tok_s": ROOF_TOK_S,
            "occupancy": occupancy,
            "parity": parity,
            "geometry_search": geo,
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

    fn bind_fused(
        enc: &metal::ComputeCommandEncoderRef,
        s0: &Buffer,
        s1: &Buffer,
        scales: &Buffer,
        scale_off: u64,
        x: &Buffer,
        y: &Buffer,
    ) {
        enc.set_buffer(0, Some(s0), 0);
        enc.set_buffer(1, Some(s1), 0);
        enc.set_buffer(2, Some(scales), scale_off);
        enc.set_buffer(3, Some(x), 0);
        enc.set_buffer(4, Some(y), 0);
    }

    fn run_one(
        queue: &CommandQueue,
        p: &ComputePipelineState,
        kernel: &str,
        rows: u32,
        bind: impl Fn(&metal::ComputeCommandEncoderRef),
    ) {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch(enc, p, kernel, rows, bind);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    fn time_n(
        queue: &CommandQueue,
        n: usize,
        build: impl Fn(&metal::ComputeCommandEncoderRef),
    ) -> (Vec<u64>, Vec<u64>) {
        let mut gpu = Vec::new();
        let mut wall = Vec::new();
        for _ in 0..n {
            let t0 = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            build(enc);
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

    fn run_parity(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
    ) -> Value {
        let mut rows_out = Vec::new();
        let pr = 16usize;
        let w = det_w(pr, GATE_COLS as usize, 41);
        let x = fill_f32(GATE_COLS as usize, 7);
        let (bc, bs) = pack_binary(&w, pr, GATE_COLS as usize);
        let w2 = det_w(pr, GATE_COLS as usize, 97);
        let (b1, s1) = pack_binary(&w2, pr, GATE_COLS as usize);
        let gpr = GATE_COLS as usize / GROUP as usize;
        let scale_rows = GATE_ROWS as usize;
        let mut sh_scales = vec![0u16; K_SHARED as usize * scale_rows * gpr];
        sh_scales[..pr * gpr].copy_from_slice(&bs);
        sh_scales[scale_rows * gpr..scale_rows * gpr + pr * gpr].copy_from_slice(&s1);
        let cpu_s = cpu_shared(&bc, &b1, &sh_scales, &x, pr, GATE_COLS as usize, scale_rows);

        let bc_buf = new_buf(device, &bc);
        let b1_buf = new_buf(device, &b1);
        let sh_buf = new_buf(device, as_u8_u16(&sh_scales));
        let x_buf = new_buf(device, as_u8_f32(&x));
        let y_buf = new_empty(device, pr * 4);
        let dots_buf = new_empty(device, K_SHARED as usize * pr * gpr * 4);

        let fused_cases = [
            "shared_binary_k2_fused_xsign_c5120_r8_tg256",
            "shared_binary_k2_fused_stream_c5120_tpr32_tg256",
            "shared_binary_k2_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k2_fused_serial_c5120",
        ];
        for kernel in fused_cases {
            let p = pipes.get(kernel).unwrap();
            run_one(queue, p, kernel, pr as u32, |enc| {
                bind_fused(enc, &bc_buf, &b1_buf, &sh_buf, 0, &x_buf, &y_buf);
            });
            let gpu = read_f32(&y_buf, pr);
            let diff = max_abs(&gpu, &cpu_s);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!(
                "parity {kernel}: max_abs={diff:.4e} {}",
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                fail(format!("parity {kernel} max_abs={diff}"));
            }
            rows_out.push(json!({
                "id": format!("fused_{kernel}"),
                "kernel": kernel,
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": true,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));
        }

        // No-op must NOT match the oracle.
        {
            let kernel = "shared_binary_k2_fused_xsign_c5120_r8_tg256_noop";
            let p = pipes.get(kernel).unwrap();
            run_one(queue, p, kernel, pr as u32, |enc| {
                bind_fused(enc, &bc_buf, &b1_buf, &sh_buf, 0, &x_buf, &y_buf);
            });
            let gpu = read_f32(&y_buf, pr);
            let diff = max_abs(&gpu, &cpu_s);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!(
                "parity {kernel}: max_abs={diff:.4e} (must_diverge) {}",
                if !ok { "OK_DIVERGE" } else { "VACUOUS" }
            );
            if ok {
                fail("noop matched the oracle; bad control is vacuous");
            }
            rows_out.push(json!({
                "id": "fused_noop_c5120",
                "kernel": kernel,
                "max_abs_diff": diff,
                "ok": false,
                "must_match": false,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));
        }

        // Two-pass BEFORE, same codes. scale_contract uses rows*gpr as k1.
        {
            let p_dots = pipes
                .get("shared_binary_k2_group_dots_c5120_g64_tpr64_tg128")
                .unwrap();
            let p_sc = pipes.get("shared_binary_k2_scale_contract_gpr80").unwrap();
            let mut packed16 = vec![0u16; K_SHARED as usize * pr * gpr];
            packed16[..pr * gpr].copy_from_slice(&bs);
            packed16[pr * gpr..].copy_from_slice(&s1);
            let sh16 = new_buf(device, as_u8_u16(&packed16));
            let (g, t) = grid_tpr64(pr as u32);
            let (sg, st) = grid_serial(pr as u32);
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(p_dots);
            enc.set_buffer(0, Some(&bc_buf), 0);
            enc.set_buffer(1, Some(&b1_buf), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&dots_buf), 0);
            set_u32(enc, 4, pr as u32);
            enc.dispatch_thread_groups(g, t);
            enc.set_compute_pipeline_state(p_sc);
            enc.set_buffer(0, Some(&sh16), 0);
            enc.set_buffer(1, Some(&dots_buf), 0);
            enc.set_buffer(2, Some(&y_buf), 0);
            set_u32(enc, 3, pr as u32);
            enc.dispatch_threads(sg, st);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu = read_f32(&y_buf, pr);
            let cpu16 = cpu_shared(&bc, &b1, &packed16, &x, pr, GATE_COLS as usize, pr);
            let diff = max_abs(&gpu, &cpu16);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!(
                "parity two_pass: max_abs={diff:.4e} {}",
                if ok { "PASS" } else { "FAIL" }
            );
            if !ok {
                fail(format!("two-pass parity max_abs={diff}"));
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

        // Down fused, 16 rows of 17408. scale_rows = DOWN_ROWS.
        {
            let prd = 16usize;
            let wd = det_w(prd, DOWN_COLS as usize, 11);
            let xd = fill_f32(DOWN_COLS as usize, 9);
            let (d0, ds0) = pack_binary(&wd, prd, DOWN_COLS as usize);
            let wd2 = det_w(prd, DOWN_COLS as usize, 13);
            let (d1, ds1) = pack_binary(&wd2, prd, DOWN_COLS as usize);
            let gprd = DOWN_COLS as usize / GROUP as usize;
            let scale_rows_d = DOWN_ROWS as usize;
            let mut shd = vec![0u16; K_SHARED as usize * scale_rows_d * gprd];
            shd[..prd * gprd].copy_from_slice(&ds0);
            shd[scale_rows_d * gprd..scale_rows_d * gprd + prd * gprd].copy_from_slice(&ds1);
            let cpu_d = cpu_shared(&d0, &d1, &shd, &xd, prd, DOWN_COLS as usize, scale_rows_d);
            let d0b = new_buf(device, &d0);
            let d1b = new_buf(device, &d1);
            let shdb = new_buf(device, as_u8_u16(&shd));
            let xdb = new_buf(device, as_u8_f32(&xd));
            let ydb = new_empty(device, prd * 4);
            for kernel in [
                "shared_binary_k2_fused_xtile_c17408_r8_tg256",
                "shared_binary_k2_fused_stream_c17408_tpr32_tg256",
                "shared_binary_k2_fused_serial_c17408",
            ] {
                let p = pipes.get(kernel).unwrap();
                run_one(queue, p, kernel, prd as u32, |enc| {
                    bind_fused(enc, &d0b, &d1b, &shdb, 0, &xdb, &ydb);
                });
                let gpu = read_f32(&ydb, prd);
                let diff = max_abs(&gpu, &cpu_d);
                let ok = diff.is_finite() && diff < PASS_TOL;
                eprintln!(
                    "parity {kernel}: max_abs={diff:.4e} {}",
                    if ok { "PASS" } else { "FAIL" }
                );
                if !ok {
                    fail(format!("parity {kernel} max_abs={diff}"));
                }
                rows_out.push(json!({
                    "id": format!("fused_{kernel}"),
                    "kernel": kernel,
                    "max_abs_diff": diff,
                    "ok": ok,
                    "must_match": true,
                    "dense_w_materialized": 0,
                    "tolerance": PASS_TOL,
                }));
            }
        }

        json!(rows_out)
    }

    fn time_kernel(
        queue: &CommandQueue,
        p: &ComputePipelineState,
        kernel: &str,
        rows: u32,
        warmup: usize,
        reps: usize,
        bind: impl Fn(&metal::ComputeCommandEncoderRef),
    ) -> (Vec<u64>, Vec<u64>) {
        let _ = time_n(queue, warmup, |enc| dispatch(enc, p, kernel, rows, |e| bind(e)));
        time_n(queue, reps, |enc| dispatch(enc, p, kernel, rows, |e| bind(e)))
    }

    fn run_geometry_search(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        warmup: usize,
        reps: usize,
    ) -> Value {
        eprintln!("geometry search (1-layer, full rows)");
        let gpr_g = GATE_COLS as usize / GROUP as usize;
        let sign_g = (GATE_ROWS as usize * GATE_COLS as usize + 7) / 8;
        let s0 = new_buf(device, &fill_u8(sign_g, 0xB000));
        let s1 = new_buf(device, &fill_u8(sign_g, 0xB100));
        let sc = new_buf(
            device,
            as_u8_u16(&fill_u16(K_SHARED as usize * GATE_ROWS as usize * gpr_g, 0xB200)),
        );
        let xg = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let yg = new_empty(device, GATE_ROWS as usize * 4);

        let mut c5120 = Vec::new();
        for kernel in FUSED_C5120 {
            let p = pipes.get(kernel).unwrap();
            let (gpu, wall) = time_kernel(queue, p, kernel, GATE_ROWS, warmup, reps, |enc| {
                bind_fused(enc, &s0, &s1, &sc, 0, &xg, &yg);
            });
            eprintln!("  {kernel} median_gpu_ns={:?}", median_u64(gpu.clone()));
            c5120.push(json!({
                "kernel": kernel,
                "gpu_ns": spread(&gpu),
                "wall_ns": spread(&wall),
                "dispatches": 1,
            }));
        }
        let winner_c5120 = c5120
            .iter()
            .min_by_key(|r| r["gpu_ns"]["median"].as_u64().unwrap_or(u64::MAX))
            .cloned()
            .unwrap();

        let gpr_d = DOWN_COLS as usize / GROUP as usize;
        let sign_d = (DOWN_ROWS as usize * DOWN_COLS as usize + 7) / 8;
        let d0 = new_buf(device, &fill_u8(sign_d, 0xC000));
        let d1 = new_buf(device, &fill_u8(sign_d, 0xC100));
        let dsc = new_buf(
            device,
            as_u8_u16(&fill_u16(K_SHARED as usize * DOWN_ROWS as usize * gpr_d, 0xC200)),
        );
        let xd = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let yd = new_empty(device, DOWN_ROWS as usize * 4);

        let mut c17408 = Vec::new();
        for kernel in FUSED_C17408 {
            let p = pipes.get(kernel).unwrap();
            let (gpu, wall) = time_kernel(queue, p, kernel, DOWN_ROWS, warmup, reps, |enc| {
                bind_fused(enc, &d0, &d1, &dsc, 0, &xd, &yd);
            });
            eprintln!("  {kernel} median_gpu_ns={:?}", median_u64(gpu.clone()));
            c17408.push(json!({
                "kernel": kernel,
                "gpu_ns": spread(&gpu),
                "wall_ns": spread(&wall),
                "dispatches": 1,
            }));
        }
        let winner_c17408 = c17408
            .iter()
            .min_by_key(|r| r["gpu_ns"]["median"].as_u64().unwrap_or(u64::MAX))
            .cloned()
            .unwrap();

        json!({
            "c5120": c5120,
            "c17408": c17408,
            "winner_c5120": winner_c5120,
            "winner_c17408": winner_c17408,
            "note": (
                "Search is 1-layer full-row GEMV. Winner is used for the 64-layer \
                 unique-scale MLP graph. q2f tpr64 is an arm, not the default."
            ),
        })
    }

    struct SharedOrg {
        s0: Buffer,
        s1: Buffer,
        scales: Buffer,
        scale_stride: usize,
        dots: Buffer,
        rows: u32,
        sign_bytes: u64,
        scale_bytes: u64,
        dots_name: &'static str,
        contract_name: &'static str,
    }

    fn build_organs(device: &Device, layers: usize) -> Vec<SharedOrg> {
        let specs = [
            (GATE_ROWS, GATE_COLS, "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128", "shared_binary_k2_scale_contract_gpr80"),
            (GATE_ROWS, GATE_COLS, "shared_binary_k2_group_dots_c5120_g64_tpr64_tg128", "shared_binary_k2_scale_contract_gpr80"),
            (DOWN_ROWS, DOWN_COLS, "shared_binary_k2_group_dots_c17408_g64_tpr64_tg128", "shared_binary_k2_scale_contract_gpr272"),
        ];
        specs
            .iter()
            .enumerate()
            .map(|(i, &(rows, cols, dots, contract))| {
                let sign_len = (rows as usize * cols as usize + 7) / 8;
                let gpr = (cols / GROUP) as usize;
                let scale_len = K_SHARED as usize * rows as usize * gpr;
                let scale_stride = align256(scale_len * 2);
                SharedOrg {
                    s0: new_buf(device, &fill_u8(sign_len, 0xB000 + i as u64)),
                    s1: new_buf(device, &fill_u8(sign_len, 0xB100 + i as u64)),
                    scales: new_buf(
                        device,
                        as_u8_u16(&fill_u16((scale_stride / 2) * layers, 0xB200 + i as u64)),
                    ),
                    scale_stride,
                    dots: new_empty(device, K_SHARED as usize * rows as usize * gpr * 4),
                    rows,
                    sign_bytes: (sign_len * 2) as u64,
                    scale_bytes: (scale_len * 2 * layers) as u64,
                    dots_name: dots,
                    contract_name: contract,
                }
            })
            .collect()
    }

    fn run_graphs(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        warmup: usize,
        reps: usize,
        layers: usize,
        win_c5120: &str,
        win_c17408: &str,
    ) -> Value {
        let built = build_organs(device, layers);
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let y_ml_gate = new_empty(device, layers * GATE_ROWS as usize * 4);
        let y_ml_down = new_empty(device, layers * DOWN_ROWS as usize * 4);

        let fused_names = [win_c5120, win_c5120, win_c17408];
        let stream_names = [
            "shared_binary_k2_fused_stream_c5120_tpr32_tg256",
            "shared_binary_k2_fused_stream_c5120_tpr32_tg256",
            "shared_binary_k2_fused_stream_c17408_tpr32_tg256",
        ];
        let serial_names = [
            "shared_binary_k2_fused_serial_c5120",
            "shared_binary_k2_fused_serial_c5120",
            "shared_binary_k2_fused_serial_c17408",
        ];
        let noop_gate = "shared_binary_k2_fused_xsign_c5120_r8_tg256_noop";

        let xy = |oi: usize| -> (&Buffer, &Buffer) {
            if oi < 2 {
                (&x_gate, &y_gate)
            } else {
                (&x_down, &y_down)
            }
        };

        let run_fused = |names: [&str; 3], n: usize| -> (Vec<u64>, Vec<u64>) {
            time_n(queue, n, |enc| {
                for layer in 0..layers {
                    for oi in 0..3 {
                        let b = &built[oi];
                        let (x, y) = xy(oi);
                        let kname = names[oi];
                        let p = pipes.get(kname).unwrap();
                        dispatch(enc, p, kname, b.rows, |e| {
                            bind_fused(
                                e,
                                &b.s0,
                                &b.s1,
                                &b.scales,
                                (layer * b.scale_stride) as u64,
                                x,
                                y,
                            );
                        });
                    }
                }
            })
        };

        let run_twopass = |n: usize| -> (Vec<u64>, Vec<u64>) {
            time_n(queue, n, |enc| {
                for layer in 0..layers {
                    for oi in 0..3 {
                        let b = &built[oi];
                        let (x, y) = xy(oi);
                        let p_dots = pipes.get(b.dots_name).unwrap();
                        dispatch(enc, p_dots, b.dots_name, b.rows, |e| {
                            e.set_buffer(0, Some(&b.s0), 0);
                            e.set_buffer(1, Some(&b.s1), 0);
                            e.set_buffer(2, Some(x), 0);
                            e.set_buffer(3, Some(&b.dots), 0);
                            set_u32(e, 4, b.rows);
                        });
                        let p_sc = pipes.get(b.contract_name).unwrap();
                        dispatch(enc, p_sc, b.contract_name, b.rows, |e| {
                            e.set_buffer(0, Some(&b.scales), (layer * b.scale_stride) as u64);
                            e.set_buffer(1, Some(&b.dots), 0);
                            e.set_buffer(2, Some(y), 0);
                            set_u32(e, 3, b.rows);
                        });
                    }
                }
            })
        };

        let run_noop = |n: usize| -> (Vec<u64>, Vec<u64>) {
            time_n(queue, n, |enc| {
                for layer in 0..layers {
                    for oi in 0..3 {
                        let b = &built[oi];
                        let (x, y) = xy(oi);
                        let kname = if oi == 0 { noop_gate } else { fused_names[oi] };
                        let p = pipes.get(kname).unwrap();
                        dispatch(enc, p, kname, b.rows, |e| {
                            bind_fused(
                                e,
                                &b.s0,
                                &b.s1,
                                &b.scales,
                                (layer * b.scale_stride) as u64,
                                x,
                                y,
                            );
                        });
                    }
                }
            })
        };

        let run_multilayer = |n: usize| -> (Vec<u64>, Vec<u64>) {
            let k_g = "shared_binary_k2_fused_xsign_layers64_c5120_r8_tg256";
            let k_d = "shared_binary_k2_fused_xtile_layers64_c17408_r8_tg256";
            time_n(queue, n, |enc| {
                for oi in 0..3 {
                    let b = &built[oi];
                    let (kname, x, y) = if oi < 2 {
                        (k_g, &x_gate, &y_ml_gate)
                    } else {
                        (k_d, &x_down, &y_ml_down)
                    };
                    let p = pipes.get(kname).unwrap();
                    dispatch(enc, p, kname, b.rows, |e| {
                        bind_fused(e, &b.s0, &b.s1, &b.scales, 0, x, y);
                    });
                }
            })
        };

        let graph = |id: &str, role: &str, dispatches: u64, kernels: Value, gpu: &[u64], wall: &[u64], extra: Value| {
            json!({
                "id": id,
                "role": role,
                "layers": layers,
                "dispatches": dispatches,
                "gpu_ns": spread(gpu),
                "wall_ns": spread(wall),
                "kernels": kernels,
                "dense_w_materialized": 0,
                "extra": extra,
            })
        };

        eprintln!("graph two_pass layers={layers}");
        let _ = run_twopass(warmup);
        let (tp_g, tp_w) = run_twopass(reps);
        eprintln!("  two_pass median_gpu_ns={:?}", median_u64(tp_g.clone()));

        eprintln!("graph fused_192 winner");
        let _ = run_fused(fused_names, warmup);
        let (fu_g, fu_w) = run_fused(fused_names, reps);
        eprintln!("  fused median_gpu_ns={:?}", median_u64(fu_g.clone()));

        eprintln!("graph fused_stream ablation");
        let _ = run_fused(stream_names, warmup);
        let (st_g, st_w) = run_fused(stream_names, reps);
        eprintln!("  stream median_gpu_ns={:?}", median_u64(st_g.clone()));

        eprintln!("graph fused_serial bad control");
        let _ = run_fused(serial_names, warmup.min(2));
        let (se_g, se_w) = run_fused(serial_names, reps);
        eprintln!("  serial median_gpu_ns={:?}", median_u64(se_g.clone()));

        eprintln!("graph fused_noop (gate noop, up/down fused)");
        let _ = run_noop(warmup.min(2));
        let (no_g, no_w) = run_noop(reps);
        eprintln!("  noop median_gpu_ns={:?}", median_u64(no_g.clone()));

        eprintln!("graph fused_multilayer_3");
        let _ = run_multilayer(warmup);
        let (ml_g, ml_w) = run_multilayer(reps);
        eprintln!("  multilayer median_gpu_ns={:?}", median_u64(ml_g.clone()));

        let sign_bytes: u64 = built.iter().map(|b| b.sign_bytes).sum();
        let scale_bytes: u64 = built.iter().map(|b| b.scale_bytes).sum();

        json!([
            graph(
                "two_pass_384",
                "before_n032",
                (layers * 3 * 2) as u64,
                json!(["group_dots + scale_contract"]),
                &tp_g,
                &tp_w,
                json!({"n032_anchor_mlp_gpu_ns": N032_TWOPASS_MLP_GPU_NS}),
            ),
            graph(
                "fused_192",
                "after_competent",
                (layers * 3) as u64,
                json!([win_c5120, win_c5120, win_c17408]),
                &fu_g,
                &fu_w,
                json!({
                    "basis_sign_bytes": sign_bytes,
                    "per_layer_scale_bytes_total": scale_bytes,
                    "mlp_payload_bytes_active_fused": sign_bytes + scale_bytes,
                    "overlap_with_stream": ranges_overlap(&fu_g, &st_g),
                    "overlap_with_serial": ranges_overlap(&fu_g, &se_g),
                    "overlap_with_noop": ranges_overlap(&fu_g, &no_g),
                    "overlap_with_twopass": ranges_overlap(&fu_g, &tp_g),
                }),
            ),
            graph(
                "fused_stream_192",
                "ablation_no_tgm",
                (layers * 3) as u64,
                json!(stream_names),
                &st_g,
                &st_w,
                json!({"overlap_with_fused": ranges_overlap(&fu_g, &st_g)}),
            ),
            graph(
                "fused_serial",
                "bad_control",
                (layers * 3) as u64,
                json!(serial_names),
                &se_g,
                &se_w,
                json!({"overlap_with_fused": ranges_overlap(&fu_g, &se_g)}),
            ),
            graph(
                "fused_noop",
                "noop_control",
                (layers * 3) as u64,
                json!([noop_gate, win_c5120, win_c17408]),
                &no_g,
                &no_w,
                json!({"overlap_with_fused": ranges_overlap(&fu_g, &no_g), "note": "gate is TGM-load-only; up/down stay fused"}),
            ),
            graph(
                "fused_multilayer_3",
                "amortize_signs_across_layers",
                3,
                json!([
                    "shared_binary_k2_fused_xsign_layers64_c5120_r8_tg256",
                    "shared_binary_k2_fused_xtile_layers64_c17408_r8_tg256",
                ]),
                &ml_g,
                &ml_w,
                json!({
                    "note": "Same x across 64 layers (harness). Real decode x changes; this arm measures sign reuse, not a sequential token.",
                    "overlap_with_fused_192": ranges_overlap(&fu_g, &ml_g),
                }),
            ),
        ])
    }
}
