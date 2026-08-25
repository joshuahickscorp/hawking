//! N036 — mixed binary/q2f MLP graphs for coherence-tax measurement.
//!
//! Injured body: independent binary g64 (1.25 bpw). Reference: q2f g64 (2.25).
//! Healing is a protected island (organ, layer band, or sparse residual) on the
//! SAME unique-weight 64-layer MLP token graph as N032, dense_w=0, >=7 reps.
//! COMPLETE_TOKEN_NS is composed in Python with the N021 non-MLP residual.
//!
//! Reuses `bytes_frontier.metal` (binary/q2f/sparse geo, group 64 as a shift).
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example binary_healing
//! ./tools/gpu_lane_lock.sh n036-binheal \
//!   workspace/ops/build/rust/release-fast/examples/binary_healing \
//!   --reps 7 --out receipts/headless/_BINARY_HEALING_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

fn usage() -> &'static str {
    "usage: binary_healing [--reps N] [--warmup N] [--layers N] [--out FILE] [--skip-unique]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("binary_healing: {message}");
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
    fail("binary_healing is Metal-only");
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
    const CSR_PCT_NUM: u32 = 1;
    const CSR_PCT_DEN: u32 = 200; // 0.5%
    const N021_COMPLETE_GPU_NS: u64 = 27_547_874;
    const MLP_ELEMENTS: u64 = 17_112_760_320;
    const PARENT_PARAMS: u64 = 26_895_998_464;
    const Q4_ATTN_F32_BYTES: u64 = 5_206_533_080;

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

    struct Csr {
        row_ptr: Vec<u32>,
        col_idx: Vec<u32>,
        corr: Vec<u16>,
    }

    fn pack_csr(rows: usize, cols: usize, seed: u32) -> Csr {
        let nnz_row = ((cols * CSR_PCT_NUM as usize) / CSR_PCT_DEN as usize).max(1);
        let mut row_ptr = vec![0u32; rows + 1];
        let mut col_idx = Vec::with_capacity(rows * nnz_row);
        let mut corr = Vec::with_capacity(rows * nnz_row);
        let mut s = u64::from(seed) | 1;
        for r in 0..rows {
            row_ptr[r] = col_idx.len() as u32;
            for _ in 0..nnz_row {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                col_idx.push((s as usize % cols) as u32);
                corr.push(f16::from_f32((((s >> 33) as i32 % 17) as f32) * 0.01).to_bits());
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

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Codec {
        Binary,
        Q2f,
    }

    fn codec_for(mode: &str, layer: usize, organ: usize, layers: usize) -> Codec {
        match mode {
            "binary" => Codec::Binary,
            "q2f" => Codec::Q2f,
            "down_q2f" => {
                if organ == 2 {
                    Codec::Q2f
                } else {
                    Codec::Binary
                }
            }
            "gate_q2f" => {
                if organ == 0 {
                    Codec::Q2f
                } else {
                    Codec::Binary
                }
            }
            "early16_q2f" => {
                if layer < 16 {
                    Codec::Q2f
                } else {
                    Codec::Binary
                }
            }
            "late16_q2f" => {
                if layer + 16 >= layers {
                    Codec::Q2f
                } else {
                    Codec::Binary
                }
            }
            _ => Codec::Binary,
        }
    }

    struct Organ {
        name: &'static str,
        rows: u32,
        cols: u32,
        binary_geo: &'static str,
        q2f_geo: &'static str,
        binary_serial: &'static str,
        sparse_geo: &'static str,
    }

    fn organs() -> [Organ; 3] {
        [
            Organ {
                name: "gate",
                rows: GATE_ROWS,
                cols: GATE_COLS,
                binary_geo: "binary_g64_matvec_geo_c5120_tpr64_tg128",
                q2f_geo: "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                binary_serial: "binary_g64_matvec_serial_c5120",
                sparse_geo: "binary_sparse_fused_geo_c5120_tpr64_tg128",
            },
            Organ {
                name: "up",
                rows: GATE_ROWS,
                cols: GATE_COLS,
                binary_geo: "binary_g64_matvec_geo_c5120_tpr64_tg128",
                q2f_geo: "q2f_g64_matvec_geo_c5120_tpr64_tg128",
                binary_serial: "binary_g64_matvec_serial_c5120",
                sparse_geo: "binary_sparse_fused_geo_c5120_tpr64_tg128",
            },
            Organ {
                name: "down",
                rows: DOWN_ROWS,
                cols: DOWN_COLS,
                binary_geo: "binary_g64_matvec_geo_c17408_tpr64_tg128",
                q2f_geo: "q2f_g64_matvec_geo_c17408_tpr64_tg128",
                binary_serial: "binary_g64_matvec_serial_c17408",
                sparse_geo: "binary_sparse_fused_geo_c17408_tpr64_tg128",
            },
        ]
    }

    struct PackedOrgan {
        codes: Buffer,
        scales: Buffer,
        code_stride: usize,
        scale_stride: usize,
        rows: u32,
        cols: u32,
        payload_bytes: u64,
        codec: Codec,
    }

    fn code_len(codec: Codec, rows: usize, cols: usize) -> usize {
        match codec {
            Codec::Binary => (rows * cols + 7) / 8,
            Codec::Q2f => (rows * cols) / 4,
        }
    }

    fn build_packed(
        device: &Device,
        org: &Organ,
        codec: Codec,
        layers: usize,
        skip_unique: bool,
        seed: u64,
    ) -> PackedOrgan {
        let gpr = (org.cols / GROUP) as usize;
        let clen = code_len(codec, org.rows as usize, org.cols as usize);
        let slen = org.rows as usize * gpr;
        let n_unique = if skip_unique { 1 } else { layers };
        let code_stride = align256(clen);
        let scale_stride = align256(slen * 2);
        let codes = fill_u8(code_stride * n_unique, seed);
        let scales = fill_u16(scale_stride / 2 * n_unique, seed ^ 0xC0FFEE);
        PackedOrgan {
            codes: new_buf(device, &codes),
            scales: new_buf(device, as_u8_u16(&scales)),
            code_stride,
            scale_stride,
            rows: org.rows,
            cols: org.cols,
            payload_bytes: (clen + slen * 2) as u64 * layers as u64,
            codec,
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

    fn kernel_for<'a>(org: &'a Organ, codec: Codec, serial: bool) -> &'a str {
        if serial {
            org.binary_serial
        } else if codec == Codec::Q2f {
            org.q2f_geo
        } else {
            org.binary_geo
        }
    }

    fn bench_island(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        mode: &str,
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
        run_serial: bool,
    ) -> Value {
        eprintln!("graph {mode} layers={layers} unique={}", !skip_unique);
        // Two packs per organ: binary and q2f. Mixed modes pick per (layer, organ).
        let bin: Vec<PackedOrgan> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                build_packed(
                    device,
                    o,
                    Codec::Binary,
                    layers,
                    skip_unique,
                    0xA000 + i as u64,
                )
            })
            .collect();
        let q2: Vec<PackedOrgan> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                build_packed(device, o, Codec::Q2f, layers, skip_unique, 0xB000 + i as u64)
            })
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let n_unique = if skip_unique { 1 } else { layers };

        let run = |serial: bool, n: usize| -> (Vec<u64>, Vec<u64>) {
            let mut gpu = Vec::new();
            let mut wall = Vec::new();
            for _ in 0..n {
                let t0 = Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let codec = codec_for(mode, layer, oi, layers);
                        if serial && codec == Codec::Q2f && oi == 2 {
                            // no q2f serial c17408 in the shader set we reuse
                            continue;
                        }
                        let packed = if codec == Codec::Q2f {
                            &q2[oi]
                        } else {
                            &bin[oi]
                        };
                        let name = kernel_for(org, codec, serial);
                        let p = pipes.get(name).unwrap_or_else(|| fail(format!("missing {name}")));
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        if serial {
                            dispatch_serial(enc, p, packed.rows, |e| {
                                bind_matvec(e, packed, x, y, layer, n_unique)
                            });
                        } else {
                            dispatch_geo(enc, p, packed.rows, |e| {
                                bind_matvec(e, packed, x, y, layer, n_unique)
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
        eprintln!("  {mode} geo median_gpu_ns={:?}", median_u64(gpu.clone()));
        let mut serial_json = Value::Null;
        if run_serial {
            let _ = run(true, warmup.min(2));
            let (sgpu, swall) = run(true, reps);
            eprintln!("  {mode} serial median_gpu_ns={:?}", median_u64(sgpu.clone()));
            serial_json = json!({
                "gpu_ns": spread(&sgpu),
                "wall_ns": spread(&swall),
                "overlap_with_geo": ranges_overlap(&gpu, &sgpu),
            });
        }

        let mut payload = 0u64;
        let mut n_bin = 0u64;
        let mut n_q2 = 0u64;
        for layer in 0..layers {
            for oi in 0..3 {
                let codec = codec_for(mode, layer, oi, layers);
                let p = if codec == Codec::Q2f {
                    n_q2 += 1;
                    &q2[oi]
                } else {
                    n_bin += 1;
                    &bin[oi]
                };
                payload += p.payload_bytes / layers as u64;
            }
        }

        json!({
            "id": mode,
            "layers": layers,
            "unique_weight_tensors": if skip_unique { 3 } else { 3 * layers },
            "dispatches": layers * 3,
            "n_binary_gemvs": n_bin,
            "n_q2f_gemvs": n_q2,
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
            "serial": serial_json,
            "mlp_payload_bytes": payload,
            "dense_w_materialized": 0,
        })
    }

    fn bench_sparse(
        device: &Device,
        queue: &CommandQueue,
        pipes: &std::collections::HashMap<&str, ComputePipelineState>,
        organs: &[Organ; 3],
        warmup: usize,
        reps: usize,
        layers: usize,
        skip_unique: bool,
    ) -> Value {
        eprintln!("graph sparse_05 layers={layers}");
        struct Packed {
            signs: Buffer,
            scales: Buffer,
            row_ptr: Buffer,
            col_idx: Buffer,
            corr: Buffer,
            sign_stride: usize,
            scale_stride: usize,
            rp_stride: usize,
            ci_stride: usize,
            corr_stride: usize,
            rows: u32,
            payload_bytes: u64,
            nnz: u64,
        }
        let built: Vec<Packed> = organs
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let rows = o.rows as usize;
                let cols = o.cols as usize;
                let sign_len = (rows * cols + 7) / 8;
                let gpr = cols / GROUP as usize;
                let scale_len = rows * gpr;
                let nnz_row = ((cols * CSR_PCT_NUM as usize) / CSR_PCT_DEN as usize).max(1);
                let nnz = rows * nnz_row;
                let n_unique = if skip_unique { 1 } else { layers };
                let sign_stride = align256(sign_len);
                let scale_stride = align256(scale_len * 2);
                let rp_stride = align256((rows + 1) * 4);
                let ci_stride = align256(nnz * 4);
                let corr_stride = align256(nnz * 2);
                let signs = fill_u8(sign_stride * n_unique, 0xC000 + i as u64);
                let scales = fill_u16(scale_stride / 2 * n_unique, 0xC100 + i as u64);
                let mut rp_all = vec![0u8; rp_stride * n_unique];
                let mut ci_all = vec![0u8; ci_stride * n_unique];
                let mut cr_all = vec![0u8; corr_stride * n_unique];
                for u in 0..n_unique {
                    let csr = pack_csr(rows, cols, 200 + i as u32 + u as u32);
                    let rp = as_u8_u32(&csr.row_ptr);
                    let ci = as_u8_u32(&csr.col_idx);
                    let cr = as_u8_u16(&csr.corr);
                    rp_all[u * rp_stride..u * rp_stride + rp.len()].copy_from_slice(rp);
                    ci_all[u * ci_stride..u * ci_stride + ci.len()].copy_from_slice(ci);
                    cr_all[u * corr_stride..u * corr_stride + cr.len()].copy_from_slice(cr);
                }
                Packed {
                    signs: new_buf(device, &signs),
                    scales: new_buf(device, as_u8_u16(&scales)),
                    row_ptr: new_buf(device, &rp_all),
                    col_idx: new_buf(device, &ci_all),
                    corr: new_buf(device, &cr_all),
                    sign_stride,
                    scale_stride,
                    rp_stride,
                    ci_stride,
                    corr_stride,
                    rows: o.rows,
                    payload_bytes: (sign_len + scale_len * 2 + (rows + 1) * 4 + nnz * 6) as u64
                        * layers as u64,
                    nnz: nnz as u64 * layers as u64,
                }
            })
            .collect();
        let x_gate = new_buf(device, as_u8_f32(&fill_f32(GATE_COLS as usize, 3)));
        let y_gate = new_empty(device, GATE_ROWS as usize * 4);
        let x_down = new_buf(device, as_u8_f32(&fill_f32(DOWN_COLS as usize, 5)));
        let y_down = new_empty(device, DOWN_ROWS as usize * 4);
        let n_unique = if skip_unique { 1 } else { layers };

        let run = |n: usize| -> (Vec<u64>, Vec<u64>) {
            let mut gpu = Vec::new();
            let mut wall = Vec::new();
            for _ in 0..n {
                let t0 = Instant::now();
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for layer in 0..layers {
                    for (oi, org) in organs.iter().enumerate() {
                        let b = &built[oi];
                        let u = layer % n_unique;
                        let p = pipes.get(org.sparse_geo).unwrap();
                        let (x, y) = if oi < 2 {
                            (&x_gate, &y_gate)
                        } else {
                            (&x_down, &y_down)
                        };
                        dispatch_geo(enc, p, b.rows, |e| {
                            e.set_buffer(0, Some(&b.signs), (u * b.sign_stride) as u64);
                            e.set_buffer(1, Some(&b.scales), (u * b.scale_stride) as u64);
                            e.set_buffer(2, Some(&b.row_ptr), (u * b.rp_stride) as u64);
                            e.set_buffer(3, Some(&b.col_idx), (u * b.ci_stride) as u64);
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
        let _ = run(warmup);
        let (gpu, wall) = run(reps);
        eprintln!(
            "  sparse_05 geo median_gpu_ns={:?}",
            median_u64(gpu.clone())
        );
        let payload: u64 = built.iter().map(|p| p.payload_bytes).sum();
        let nnz: u64 = built.iter().map(|p| p.nnz).sum();
        json!({
            "id": "sparse_05",
            "layers": layers,
            "unique_weight_tensors": if skip_unique { 3 } else { 3 * layers },
            "dispatches": layers * 3,
            "nnz": nnz,
            "nnz_frac": (CSR_PCT_NUM as f64) / (CSR_PCT_DEN as f64),
            "gpu_ns": spread(&gpu),
            "wall_ns": spread(&wall),
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
        let pr = 16usize;
        let w = det_w(pr, GATE_COLS as usize, 41);
        let x = fill_f32(GATE_COLS as usize, 7);
        let (bc, bs) = pack_binary(&w, pr, GATE_COLS as usize);
        let cpu_b = cpu_binary(&bc, &bs, &x, pr, GATE_COLS as usize);
        let (qc, qd) = pack_q2f(&w, pr, GATE_COLS as usize);
        let cpu_q = cpu_q2f(&qc, &qd, &x, pr, GATE_COLS as usize);
        let csr = pack_csr(pr, GATE_COLS as usize, 99);
        let cpu_r = cpu_residual(&bc, &bs, &csr, &x, pr, GATE_COLS as usize);

        let bc_buf = new_buf(device, &bc);
        let bs_buf = new_buf(device, as_u8_u16(&bs));
        let qc_buf = new_buf(device, &qc);
        let qd_buf = new_buf(device, as_u8_u16(&qd));
        let x_buf = new_buf(device, as_u8_f32(&x));
        let rp_buf = new_buf(device, as_u8_u32(&csr.row_ptr));
        let ci_buf = new_buf(device, as_u8_u32(&csr.col_idx));
        let cr_buf = new_buf(device, as_u8_u16(&csr.corr));
        let y_buf = new_empty(device, pr * 4);

        let cases: &[(&str, &str, Box<dyn Fn(&metal::ComputeCommandEncoderRef)>, &[f32], bool)] = &[
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
                "parity {id}: max_abs={diff:.4e} {}",
                if ok { "PASS" } else { "DIFF" }
            );
            if *must_match && !ok {
                fail(format!("parity {id} max_abs={diff}"));
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
        }
        json!(rows_out)
    }

    pub fn run(args: Args) {
        let t_all = Instant::now();
        let device = Device::system_default().unwrap_or_else(|| fail("no Metal GPU"));
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        eprintln!("binary_healing: compile shader");
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .unwrap_or_else(|e| fail(format!("shader compile: {e}")));
        let compile_s = t0.elapsed().as_secs_f64();
        eprintln!("  compiled in {compile_s:.3}s");

        let names = [
            "binary_g64_matvec_geo_c5120_tpr64_tg128",
            "binary_g64_matvec_geo_c17408_tpr64_tg128",
            "binary_g64_matvec_serial_c5120",
            "binary_g64_matvec_serial_c17408",
            "binary_sparse_fused_geo_c5120_tpr64_tg128",
            "binary_sparse_fused_geo_c17408_tpr64_tg128",
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
        let organs = organs();
        let mut graphs = Vec::new();
        for (mode, serial) in [
            ("binary", true),
            ("q2f", false),
            ("down_q2f", true),
            ("gate_q2f", false),
            ("early16_q2f", false),
            ("late16_q2f", false),
        ] {
            graphs.push(bench_island(
                &device,
                &queue,
                &pipes,
                &organs,
                mode,
                args.warmup,
                args.reps,
                args.layers,
                args.skip_unique,
                serial,
            ));
        }
        graphs.push(bench_sparse(
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
            "schema": "hawking.headless.binary_healing.raw.v1",
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
