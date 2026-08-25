//! N035 — COMPLETE_TOKEN_NS of the N033 fused shared-basis operator at K=2/4/8/16.
//!
//! K=2 uses the N033-winning 2-buffer tpr64 kernels. K=4/8/16 use concatenated
//! sign planes on the same geometry. 64-layer unique-scale MLP graph (192 GEMVs).
//! dense_w = 0.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example shared_basis_coherent
//! ./tools/gpu_lane_lock.sh n035-sharedcoherent \
//!   workspace/ops/build/rust/release-fast/examples/shared_basis_coherent \
//!   --reps 7 --out receipts/headless/_SHARED_BASIS_COHERENT_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

fn usage() -> &'static str {
    "usage: shared_basis_coherent [--reps N] [--warmup N] [--layers N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("shared_basis_coherent: {message}");
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
    fail("shared_basis_coherent is Metal-only");
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

    const SHADER: &str = include_str!("../shaders/shared_basis_kernel.metal");
    const PASS_TOL: f32 = 2e-2;
    const GATE_ROWS: u32 = 17408;
    const GATE_COLS: u32 = 5120;
    const DOWN_ROWS: u32 = 5120;
    const DOWN_COLS: u32 = 17408;
    const GROUP: u32 = 64;
    const PLANE: usize = (GATE_ROWS as usize * GATE_COLS as usize) / 8; // 11_141_120
    const KST: usize = 1_392_640; // rows*gpr for both organs
    const N021_COMPLETE_GPU_NS: u64 = 27_547_874;
    const N032_Q2F_MLP_GPU_NS: u64 = 15_738_249;
    const N033_K2_MLP_GPU_NS: u64 = 12_745_000;

    fn as_u8_u16(v: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
    }
    fn as_u8_f32(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
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

    fn cpu_shared_k(
        planes: &[Vec<u8>],
        scales: &[u16],
        x: &[f32],
        rows: usize,
        cols: usize,
        scale_rows: usize,
        k: usize,
    ) -> Vec<f32> {
        let gpr = cols / GROUP as usize;
        let mut y = vec![0.0f32; rows];
        for ki in 0..k {
            let yk = cpu_binary(
                &planes[ki],
                &scales[ki * scale_rows * gpr..(ki * scale_rows * gpr + rows * gpr)],
                x,
                rows,
                cols,
            );
            for (a, b) in y.iter_mut().zip(yk) {
                *a += b;
            }
        }
        y
    }

    fn grid_tpr64(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows / 2), 1, 1), MTLSize::new(128, 1, 1))
    }
    fn grid_tpr32(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows / 8), 1, 1), MTLSize::new(256, 1, 1))
    }
    fn grid_serial(rows: u32) -> (MTLSize, MTLSize) {
        (MTLSize::new(u64::from(rows), 1, 1), MTLSize::new(256, 1, 1))
    }

    fn grid_for(kernel: &str, rows: u32) -> (MTLSize, MTLSize) {
        if kernel.contains("serial") {
            grid_serial(rows)
        } else if kernel.contains("tpr64") {
            grid_tpr64(rows)
        } else {
            grid_tpr32(rows)
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
        if kernel.contains("serial") {
            enc.dispatch_threads(g, t);
        } else {
            enc.dispatch_thread_groups(g, t);
        }
    }

    fn compile(device: &Device, src: &str) -> metal::Library {
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        eprintln!("shared_basis_coherent: compile fused");
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(src, &opts)
            .unwrap_or_else(|e| fail(format!("shader compile: {e}")));
        eprintln!("  compiled in {:.3}s", t0.elapsed().as_secs_f64());
        lib
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

    pub fn run(args: Args) {
        let t_all = Instant::now();
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal GPU"));
        let queue = device.new_command_queue();
        let lib = compile(&device, SHADER);

        let names = [
            "shared_binary_k2_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k2_fused_stream_c17408_tpr64_tg128",
            "shared_binary_k4_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k4_fused_stream_c17408_tpr64_tg128",
            "shared_binary_k8_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k8_fused_stream_c17408_tpr64_tg128",
            "shared_binary_k8_fused_stream_c5120_tpr32_tg256",
            "shared_binary_k8_fused_serial_c5120",
            "shared_binary_k8_fused_serial_c17408",
            "shared_binary_k8_fused_stream_c5120_tpr64_tg128_noop",
            "shared_binary_k16_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k16_fused_stream_c17408_tpr64_tg128",
        ];
        let mut pipes = std::collections::HashMap::new();
        let mut occupancy = Vec::new();
        for n in names {
            let p = pipe(&device, &lib, n);
            occupancy.push(json!({
                "kernel": n,
                "max_total_threads_per_threadgroup": p.max_total_threads_per_threadgroup(),
                "thread_execution_width": p.thread_execution_width(),
                "static_threadgroup_memory_length": p.static_threadgroup_memory_length(),
            }));
            pipes.insert(n, p);
        }

        let parity = run_parity(&device, &queue, &pipes);
        let graphs = run_graphs(&device, &queue, &pipes, args.warmup, args.reps, args.layers);

        let doc = json!({
            "schema": "hawking.headless.shared_basis_coherent.raw.v1",
            "git_head": git_head(),
            "device": device.name().to_string(),
            "reps": args.reps,
            "warmup": args.warmup,
            "layers": args.layers,
            "dense_w_materialized": 0,
            "n021_complete_gpu_ns_median": N021_COMPLETE_GPU_NS,
            "n032_q2f_mlp_gpu_ns_median": N032_Q2F_MLP_GPU_NS,
            "n033_k2_mlp_gpu_ns_median": N033_K2_MLP_GPU_NS,
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

    fn concat_planes(planes: &[Vec<u8>], plane: usize) -> Vec<u8> {
        let mut out = vec![0u8; planes.len() * plane];
        for (k, p) in planes.iter().enumerate() {
            out[k * plane..k * plane + p.len()].copy_from_slice(p);
        }
        out
    }

    fn run_parity(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
    ) -> Value {
        let mut rows_out = Vec::new();
        let pr = 16usize;
        let cols = GATE_COLS as usize;
        let x = fill_f32(cols, 7);
        let gpr = cols / GROUP as usize;
        let scale_rows = GATE_ROWS as usize;

        for k in [4usize, 8, 16] {
            let mut planes = Vec::new();
            let mut sh_scales = vec![0u16; k * scale_rows * gpr];
            for ki in 0..k {
                let w = det_w(pr, cols, 41 + (ki as u32) * 17);
                let (bits, sc) = pack_binary(&w, pr, cols);
                planes.push(bits);
                sh_scales[ki * scale_rows * gpr..ki * scale_rows * gpr + pr * gpr]
                    .copy_from_slice(&sc);
            }
            let cpu = cpu_shared_k(&planes, &sh_scales, &x, pr, cols, scale_rows, k);
            let packed = concat_planes(&planes, PLANE);
            let sbuf = new_buf(device, &packed);
            let scbuf = new_buf(device, as_u8_u16(&sh_scales));
            let xbuf = new_buf(device, as_u8_f32(&x));
            let ybuf = new_empty(device, pr * 4);
            let kernel = format!("shared_binary_k{k}_fused_stream_c5120_tpr64_tg128");
            let p = pipes.get(kernel.as_str()).unwrap();
            run_one(queue, p, &kernel, pr as u32, |enc| {
                enc.set_buffer(0, Some(&sbuf), 0);
                enc.set_buffer(1, Some(&scbuf), 0);
                enc.set_buffer(2, Some(&xbuf), 0);
                enc.set_buffer(3, Some(&ybuf), 0);
            });
            let gpu = read_f32(&ybuf, pr);
            let diff = max_abs(&gpu, &cpu);
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
                "k": k,
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": true,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));
        }

        // K=8 serial must match; noop must diverge.
        {
            let k = 8usize;
            let mut planes = Vec::new();
            let mut sh_scales = vec![0u16; k * scale_rows * gpr];
            for ki in 0..k {
                let w = det_w(pr, cols, 41 + (ki as u32) * 17);
                let (bits, sc) = pack_binary(&w, pr, cols);
                planes.push(bits);
                sh_scales[ki * scale_rows * gpr..ki * scale_rows * gpr + pr * gpr]
                    .copy_from_slice(&sc);
            }
            let cpu = cpu_shared_k(&planes, &sh_scales, &x, pr, cols, scale_rows, k);
            let packed = concat_planes(&planes, PLANE);
            let sbuf = new_buf(device, &packed);
            let scbuf = new_buf(device, as_u8_u16(&sh_scales));
            let xbuf = new_buf(device, as_u8_f32(&x));
            let ybuf = new_empty(device, pr * 4);
            let kernel = "shared_binary_k8_fused_serial_c5120";
            let p = pipes.get(kernel).unwrap();
            run_one(queue, p, kernel, pr as u32, |enc| {
                enc.set_buffer(0, Some(&sbuf), 0);
                enc.set_buffer(1, Some(&scbuf), 0);
                enc.set_buffer(2, Some(&xbuf), 0);
                enc.set_buffer(3, Some(&ybuf), 0);
            });
            let gpu = read_f32(&ybuf, pr);
            let diff = max_abs(&gpu, &cpu);
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
                "k": 8,
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": true,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));

            let kernel = "shared_binary_k8_fused_stream_c5120_tpr64_tg128_noop";
            let p = pipes.get(kernel).unwrap();
            run_one(queue, p, kernel, pr as u32, |enc| {
                enc.set_buffer(0, Some(&sbuf), 0);
                enc.set_buffer(1, Some(&scbuf), 0);
                enc.set_buffer(2, Some(&xbuf), 0);
                enc.set_buffer(3, Some(&ybuf), 0);
            });
            let gpu = read_f32(&ybuf, pr);
            let diff = max_abs(&gpu, &cpu);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!(
                "parity {kernel}: max_abs={diff:.4e} (must_diverge) {}",
                if !ok { "OK_DIVERGE" } else { "VACUOUS" }
            );
            if ok {
                fail("noop matched the oracle; bad control is vacuous");
            }
            rows_out.push(json!({
                "id": "fused_noop_k8_c5120",
                "kernel": kernel,
                "k": 8,
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": false,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));
        }

        // down_proj K=8 tpr64, 16 rows.
        {
            let k = 8usize;
            let pr = 16usize;
            let cols = DOWN_COLS as usize;
            let rows_full = DOWN_ROWS as usize;
            let gpr = cols / GROUP as usize;
            let x = fill_f32(cols, 11);
            let mut planes = Vec::new();
            let mut sh_scales = vec![0u16; k * rows_full * gpr];
            for ki in 0..k {
                let w = det_w(pr, cols, 97 + (ki as u32) * 13);
                let (bits, sc) = pack_binary(&w, pr, cols);
                planes.push(bits);
                sh_scales[ki * rows_full * gpr..ki * rows_full * gpr + pr * gpr].copy_from_slice(&sc);
            }
            let cpu = cpu_shared_k(&planes, &sh_scales, &x, pr, cols, rows_full, k);
            let packed = concat_planes(&planes, PLANE);
            let sbuf = new_buf(device, &packed);
            let scbuf = new_buf(device, as_u8_u16(&sh_scales));
            let xbuf = new_buf(device, as_u8_f32(&x));
            let ybuf = new_empty(device, pr * 4);
            let kernel = "shared_binary_k8_fused_stream_c17408_tpr64_tg128";
            let p = pipes.get(kernel).unwrap();
            run_one(queue, p, kernel, pr as u32, |enc| {
                enc.set_buffer(0, Some(&sbuf), 0);
                enc.set_buffer(1, Some(&scbuf), 0);
                enc.set_buffer(2, Some(&xbuf), 0);
                enc.set_buffer(3, Some(&ybuf), 0);
            });
            let gpu = read_f32(&ybuf, pr);
            let diff = max_abs(&gpu, &cpu);
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
                "k": 8,
                "max_abs_diff": diff,
                "ok": ok,
                "must_match": true,
                "dense_w_materialized": 0,
                "tolerance": PASS_TOL,
            }));
        }

        json!(rows_out)
    }

    struct OrganK {
        signs: Buffer,
        scales: Buffer,
        scale_stride: usize,
        rows: u32,
        kernel: &'static str,
        sign_bytes: u64,
        scale_bytes: u64,
    }

    fn run_graphs(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        warmup: usize,
        reps: usize,
        layers: usize,
    ) -> Value {
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);

        // K=2: existing two-buffer kernels (N033 winner).
        let k2_s0g = new_buf(device, &fill_u8(PLANE, 0xB000));
        let k2_s1g = new_buf(device, &fill_u8(PLANE, 0xB100));
        let k2_scg = new_buf(
            device,
            as_u8_u16(&fill_u16(2 * GATE_ROWS as usize * 80 * layers, 0xB200)),
        );
        let k2_s0d = new_buf(device, &fill_u8(PLANE, 0xC000));
        let k2_s1d = new_buf(device, &fill_u8(PLANE, 0xC100));
        let k2_scd = new_buf(
            device,
            as_u8_u16(&fill_u16(2 * DOWN_ROWS as usize * 272 * layers, 0xC200)),
        );
        let k2_gate = "shared_binary_k2_fused_stream_c5120_tpr64_tg128";
        let k2_down = "shared_binary_k2_fused_stream_c17408_tpr64_tg128";
        let k2_g_stride = 2 * GATE_ROWS as usize * 80 * 2;
        let k2_d_stride = 2 * DOWN_ROWS as usize * 272 * 2;

        let run_k2 = |n: usize| -> (Vec<u64>, Vec<u64>) {
            time_n(queue, n, |enc| {
                for layer in 0..layers {
                    let p = pipes.get(k2_gate).unwrap();
                    dispatch(enc, p, k2_gate, GATE_ROWS, |e| {
                        e.set_buffer(0, Some(&k2_s0g), 0);
                        e.set_buffer(1, Some(&k2_s1g), 0);
                        e.set_buffer(2, Some(&k2_scg), (layer * k2_g_stride) as u64);
                        e.set_buffer(3, Some(&x_gate), 0);
                        e.set_buffer(4, Some(&y_gate), 0);
                    });
                    dispatch(enc, p, k2_gate, GATE_ROWS, |e| {
                        e.set_buffer(0, Some(&k2_s0g), 0);
                        e.set_buffer(1, Some(&k2_s1g), 0);
                        e.set_buffer(2, Some(&k2_scg), (layer * k2_g_stride) as u64);
                        e.set_buffer(3, Some(&x_gate), 0);
                        e.set_buffer(4, Some(&y_gate), 0);
                    });
                    let p = pipes.get(k2_down).unwrap();
                    dispatch(enc, p, k2_down, DOWN_ROWS, |e| {
                        e.set_buffer(0, Some(&k2_s0d), 0);
                        e.set_buffer(1, Some(&k2_s1d), 0);
                        e.set_buffer(2, Some(&k2_scd), (layer * k2_d_stride) as u64);
                        e.set_buffer(3, Some(&x_down), 0);
                        e.set_buffer(4, Some(&y_down), 0);
                    });
                }
            })
        };

        let build_k = |k: usize, kg: &'static str, kd: &'static str| -> Vec<OrganK> {
            let specs = [
                (GATE_ROWS, 80usize, kg),
                (GATE_ROWS, 80usize, kg),
                (DOWN_ROWS, 272usize, kd),
            ];
            specs
                .iter()
                .enumerate()
                .map(|(i, &(rows, gpr, kernel))| {
                    let scale_len = k * rows as usize * gpr;
                    let scale_stride = (scale_len * 2 + 255) & !255;
                    OrganK {
                        signs: new_buf(device, &fill_u8(k * PLANE, 0xD000 + (k * 16 + i) as u64)),
                        scales: new_buf(
                            device,
                            as_u8_u16(&fill_u16((scale_stride / 2) * layers, 0xE000 + i as u64)),
                        ),
                        scale_stride,
                        rows,
                        kernel,
                        sign_bytes: (k * PLANE) as u64,
                        scale_bytes: (scale_len * 2 * layers) as u64,
                    }
                })
                .collect()
        };

        let run_concat = |organs: &[OrganK], n: usize| -> (Vec<u64>, Vec<u64>) {
            time_n(queue, n, |enc| {
                for layer in 0..layers {
                    for (oi, b) in organs.iter().enumerate() {
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        let p = pipes.get(b.kernel).unwrap();
                        dispatch(enc, p, b.kernel, b.rows, |e| {
                            e.set_buffer(0, Some(&b.signs), 0);
                            e.set_buffer(1, Some(&b.scales), (layer * b.scale_stride) as u64);
                            e.set_buffer(2, Some(x), 0);
                            e.set_buffer(3, Some(y), 0);
                        });
                    }
                }
            })
        };

        let mut graphs = Vec::new();

        eprintln!("graph fused_k2 layers={layers}");
        let _ = run_k2(warmup);
        let (g, w) = run_k2(reps);
        eprintln!("  k2 median_gpu_ns={:?}", median_u64(g.clone()));
        graphs.push(json!({
            "id": "fused_k2_192",
            "k": 2,
            "role": "n033_winner_replay",
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&g),
            "wall_ns": spread(&w),
            "kernels": [k2_gate, k2_gate, k2_down],
            "dense_w_materialized": 0,
        }));
        let k2_gpu = g;

        let k4_org = build_k(
            4,
            "shared_binary_k4_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k4_fused_stream_c17408_tpr64_tg128",
        );
        eprintln!("graph fused_k4 layers={layers}");
        let _ = run_concat(&k4_org, warmup);
        let (g, w) = run_concat(&k4_org, reps);
        eprintln!("  k4 median_gpu_ns={:?}", median_u64(g.clone()));
        let k4_sign: u64 = k4_org.iter().map(|b| b.sign_bytes).sum();
        let k4_scale: u64 = k4_org.iter().map(|b| b.scale_bytes).sum();
        graphs.push(json!({
            "id": "fused_k4_192",
            "k": 4,
            "role": "k_sweep",
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&g),
            "wall_ns": spread(&w),
            "kernels": [
                "shared_binary_k4_fused_stream_c5120_tpr64_tg128",
                "shared_binary_k4_fused_stream_c17408_tpr64_tg128",
            ],
            "basis_sign_bytes": k4_sign,
            "per_layer_scale_bytes_total": k4_scale,
            "dense_w_materialized": 0,
        }));

        let k8_org = build_k(
            8,
            "shared_binary_k8_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k8_fused_stream_c17408_tpr64_tg128",
        );
        eprintln!("graph fused_k8 layers={layers}");
        let _ = run_concat(&k8_org, warmup);
        let (g8, w8) = run_concat(&k8_org, reps);
        eprintln!("  k8 median_gpu_ns={:?}", median_u64(g8.clone()));
        let k8_sign: u64 = k8_org.iter().map(|b| b.sign_bytes).sum();
        let k8_scale: u64 = k8_org.iter().map(|b| b.scale_bytes).sum();

        // K=8 tpr32 ablation (gate uses tpr32, down stays tpr64).
        let k8_tpr32_org = build_k(
            8,
            "shared_binary_k8_fused_stream_c5120_tpr32_tg256",
            "shared_binary_k8_fused_stream_c17408_tpr64_tg128",
        );
        eprintln!("graph fused_k8_tpr32_ablation");
        let _ = run_concat(&k8_tpr32_org, warmup.min(2));
        let (g32, w32) = run_concat(&k8_tpr32_org, reps);

        // serial / noop controls on K=8.
        let run_named = |names: [&str; 3], n: usize| -> (Vec<u64>, Vec<u64>) {
            time_n(queue, n, |enc| {
                for layer in 0..layers {
                    for (oi, b) in k8_org.iter().enumerate() {
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        let kname = names[oi];
                        let p = pipes.get(kname).unwrap();
                        dispatch(enc, p, kname, b.rows, |e| {
                            e.set_buffer(0, Some(&b.signs), 0);
                            e.set_buffer(1, Some(&b.scales), (layer * b.scale_stride) as u64);
                            e.set_buffer(2, Some(x), 0);
                            e.set_buffer(3, Some(y), 0);
                        });
                    }
                }
            })
        };
        eprintln!("graph fused_k8_serial");
        let serial_names = [
            "shared_binary_k8_fused_serial_c5120",
            "shared_binary_k8_fused_serial_c5120",
            "shared_binary_k8_fused_serial_c17408",
        ];
        let _ = run_named(serial_names, warmup.min(1));
        let (gse, wse) = run_named(serial_names, reps);
        eprintln!("  serial median_gpu_ns={:?}", median_u64(gse.clone()));

        eprintln!("graph fused_k8_noop");
        let noop_names = [
            "shared_binary_k8_fused_stream_c5120_tpr64_tg128_noop",
            "shared_binary_k8_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k8_fused_stream_c17408_tpr64_tg128",
        ];
        let _ = run_named(noop_names, warmup.min(2));
        let (gno, wno) = run_named(noop_names, reps);

        graphs.push(json!({
            "id": "fused_k8_192",
            "k": 8,
            "role": "hypothesized_coherent_point",
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&g8),
            "wall_ns": spread(&w8),
            "kernels": [
                "shared_binary_k8_fused_stream_c5120_tpr64_tg128",
                "shared_binary_k8_fused_stream_c17408_tpr64_tg128",
            ],
            "basis_sign_bytes": k8_sign,
            "per_layer_scale_bytes_total": k8_scale,
            "overlap_with_serial": ranges_overlap(&g8, &gse),
            "overlap_with_noop": ranges_overlap(&g8, &gno),
            "overlap_with_tpr32": ranges_overlap(&g8, &g32),
            "overlap_with_k2": ranges_overlap(&g8, &k2_gpu),
            "dense_w_materialized": 0,
        }));
        graphs.push(json!({
            "id": "fused_k8_tpr32_ablation",
            "k": 8,
            "role": "geometry_ablation",
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&g32),
            "wall_ns": spread(&w32),
            "dense_w_materialized": 0,
        }));
        graphs.push(json!({
            "id": "fused_k8_serial",
            "k": 8,
            "role": "bad_control",
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&gse),
            "wall_ns": spread(&wse),
            "overlap_with_fused": ranges_overlap(&g8, &gse),
            "dense_w_materialized": 0,
        }));
        graphs.push(json!({
            "id": "fused_k8_noop",
            "k": 8,
            "role": "noop_control",
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&gno),
            "wall_ns": spread(&wno),
            "overlap_with_fused": ranges_overlap(&g8, &gno),
            "dense_w_materialized": 0,
        }));

        let k16_org = build_k(
            16,
            "shared_binary_k16_fused_stream_c5120_tpr64_tg128",
            "shared_binary_k16_fused_stream_c17408_tpr64_tg128",
        );
        eprintln!("graph fused_k16 layers={layers}");
        let _ = run_concat(&k16_org, warmup);
        let (g, w) = run_concat(&k16_org, reps);
        eprintln!("  k16 median_gpu_ns={:?}", median_u64(g.clone()));
        let k16_sign: u64 = k16_org.iter().map(|b| b.sign_bytes).sum();
        let k16_scale: u64 = k16_org.iter().map(|b| b.scale_bytes).sum();
        graphs.push(json!({
            "id": "fused_k16_192",
            "k": 16,
            "role": "k_sweep_above_q2f",
            "layers": layers,
            "dispatches": layers * 3,
            "gpu_ns": spread(&g),
            "wall_ns": spread(&w),
            "basis_sign_bytes": k16_sign,
            "per_layer_scale_bytes_total": k16_scale,
            "dense_w_materialized": 0,
        }));

        json!(graphs)
    }
}
