//! Isolated in-register reconstruction cost on Qwen3.8 dense shapes.
//!
//! GPU time authority: MTLCommandBuffer GPUStartTime/GPUEndTime after wait.

use metal::objc::{msg_send, sel, sel_impl};
use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use serde_json::{json, Value};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const SHADER: &str = include_str!("../recon.metal");
const HONEST_CEILING_GBPS: f64 = 411.51;
const QWEN38_ACHIEVED_GBPS: f64 = 406.2;
const WARMUP: usize = 2;
const REPS: usize = 5;
const CONTROL_TGS: u32 = 60;
const CONTROL_TPTG: u32 = 256;
const CONTROL_ITERS: u32 = 4096;
const CONTROL_BYTES: u64 = 256 * 1024 * 1024;

fn gpu_ns(cmd: &metal::CommandBufferRef) -> Result<u64, Box<dyn Error>> {
    let start: f64 = unsafe { msg_send![cmd, GPUStartTime] };
    let end: f64 = unsafe { msg_send![cmd, GPUEndTime] };
    if end > start && start > 0.0 {
        Ok(((end - start) * 1e9) as u64)
    } else {
        Err("missing GPU timestamp".into())
    }
}

fn median(v: &[u64]) -> u64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

fn read_bytes(p: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(fs::read(p)?)
}

fn read_f32(p: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let b = fs::read(p)?;
    if b.len() % 4 != 0 {
        return Err(format!("{p}: not f32-aligned").into());
    }
    let n = b.len() / 4;
    let mut o = vec![0f32; n];
    o.copy_from_slice(bytemuck_cast_f32(&b));
    Ok(o)
}

fn bytemuck_cast_f32(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        num += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    let d = na.sqrt() * nb.sqrt();
    if d <= 1e-30 {
        1.0
    } else {
        num / d
    }
}

struct Lib {
    device: Device,
    queue: metal::CommandQueue,
    lib: metal::Library,
}

impl Lib {
    fn new() -> Result<Self, Box<dyn Error>> {
        let device = Device::system_default().ok_or("no Metal device")?;
        let opts = CompileOptions::new();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("shader compile: {e}"))?;
        Ok(Self {
            queue: device.new_command_queue(),
            device,
            lib,
        })
    }

    fn pipe(&self, name: &str) -> Result<ComputePipelineState, Box<dyn Error>> {
        let f = self
            .lib
            .get_function(name, None)
            .map_err(|e| format!("{name}: {e}"))?;
        self.device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}").into())
    }

    fn buf(&self, bytes: &[u8]) -> metal::Buffer {
        self.device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn buf_len(&self, n: usize) -> metal::Buffer {
        self.device
            .new_buffer(n as u64, MTLResourceOptions::StorageModeShared)
    }

    fn set_u32(enc: &metal::ComputeCommandEncoderRef, idx: u64, v: u32) {
        enc.set_bytes(idx, 4, &v as *const u32 as *const _);
    }

    fn time_once(
        &self,
        inner: u32,
        encode: impl Fn(&metal::CommandBufferRef),
    ) -> Result<(u64, u64), Box<dyn Error>> {
        let cmd = self.queue.new_command_buffer();
        for _ in 0..inner {
            encode(cmd);
        }
        let t0 = std::time::Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let cpu_ns = t0.elapsed().as_nanos() as u64;
        let g = gpu_ns(cmd)?;
        Ok((g / inner.max(1) as u64, cpu_ns / inner.max(1) as u64))
    }

    fn time_reps(
        &self,
        encode: impl Fn(&metal::CommandBufferRef),
    ) -> Result<(u64, Vec<u64>, u64), Box<dyn Error>> {
        // Single-dispatch GPU timestamps are launch-dominated (~15 µs) on
        // this shape. Authority is a 16-dispatch command buffer, per-iter.
        for _ in 0..WARMUP {
            let _ = self.time_once(1, &encode)?;
        }
        let mut ns = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            ns.push(self.time_once(1, &encode)?.0);
        }
        let mut x16 = Vec::with_capacity(3);
        let mut x16_cpu = Vec::with_capacity(3);
        for _ in 0..3 {
            let (g, c) = self.time_once(16, &encode)?;
            x16.push(g);
            x16_cpu.push(c);
        }
        let ax = median(&x16);
        eprintln!(
            "    single={}us  x16_med={}us x16_cpu={}us",
            median(&ns) / 1000,
            ax / 1000,
            median(&x16_cpu) / 1000
        );
        Ok((ax, ns, median(&ns)))
    }
}

fn tpr64_grid(rows: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    let tg = 128u32;
    let grid = rows.div_ceil(2).saturating_mul(tg).max(tg);
    ((grid, 1, 1), (tg, 1, 1))
}

fn tg256_grid(rows: u32) -> ((u32, u32, u32), (u32, u32, u32)) {
    ((rows.saturating_mul(256).max(256), 1, 1), (256, 1, 1))
}

fn dispatch(
    enc: &metal::ComputeCommandEncoderRef,
    pipe: &ComputePipelineState,
    grid: (u32, u32, u32),
    tg: (u32, u32, u32),
) {
    enc.set_compute_pipeline_state(pipe);
    enc.dispatch_threads(
        MTLSize::new(grid.0 as u64, grid.1 as u64, grid.2 as u64),
        MTLSize::new(tg.0 as u64, tg.1 as u64, tg.2 as u64),
    );
}

fn read_out(buf: &metal::Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
}

fn zero_out(buf: &metal::Buffer, n: usize) {
    unsafe {
        std::ptr::write_bytes(buf.contents() as *mut u8, 0, n * 4);
    }
}

fn variant(
    name: &str,
    kernel: &str,
    bytes: u64,
    med: u64,
    reps: &[u64],
    y: Option<(&[f32], &[f32])>,
    note: &str,
    single_ns: u64,
) -> Value {
    let gbps = if med == 0 {
        0.0
    } else {
        (bytes as f64) / (med as f64)
    };
    let floor = (bytes as f64) / QWEN38_ACHIEVED_GBPS;
    let excess = (med as f64 - floor).max(0.0);
    let penalty = if floor <= 0.0 { 0.0 } else { med as f64 / floor };
    let corr = if let Some((got, exp)) = y {
        let n = got.len().min(exp.len());
        json!({
            "max_abs": max_abs(&got[..n], &exp[..n]),
            "cosine": cosine(&got[..n], &exp[..n]),
            "n": n,
        })
    } else {
        json!(null)
    };
    json!({
        "name": name,
        "kernel": kernel,
        "traffic_bytes": bytes,
        "median_gpu_ns": med,
        "timing_authority": "16-dispatch command buffer, per-iter GPUEndTime-GPUStartTime",
        "single_dispatch_gpu_ns": single_ns,
        "single_dispatch_gpu_ns_reps": reps,
        "packed_gbps": gbps,
        "frac_of_honest_ceiling": gbps / HONEST_CEILING_GBPS,
        "frac_of_qwen38_achieved": gbps / QWEN38_ACHIEVED_GBPS,
        "bandwidth_floor_ns": floor,
        "recon_excess_ns": excess,
        "recon_penalty_vs_bandwidth": penalty,
        "correctness": corr,
        "note": note,
    })
}

fn run_organ(lib: &Lib, organ: &Value, skip_f32: bool) -> Result<Value, Box<dyn Error>> {
    let name = organ["name"].as_str().unwrap();
    let rows = organ["rows"].as_u64().unwrap() as u32;
    let cols = organ["cols"].as_u64().unwrap() as u32;
    let x = read_f32(organ["x"].as_str().unwrap())?;
    let x_buf = lib.buf(unsafe {
        std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4)
    });
    let y_buf = lib.buf_len((rows as usize) * 4);
    let codecs = organ["codecs"].as_object().unwrap();
    let mut results = Vec::new();

    let pipe_f32 = lib.pipe("disc_f32_tpr64")?;
    let pipe_nibble = lib.pipe("disc_q4_nibble_tpr64")?;
    let pipe_uni = lib.pipe("disc_uniform_bits_tpr64")?;
    let pipe_uni256 = lib.pipe("disc_uniform_bits_tg256")?;
    let pipe_uni_ser = lib.pipe("disc_uniform_bits_serial")?;
    let pipe_bin = lib.pipe("disc_binary_tpr64")?;
    let pipe_bin256 = lib.pipe("disc_binary_tg256")?;
    let pipe_ter = lib.pipe("disc_ternary_tpr64")?;
    let pipe_add = lib.pipe("disc_additive_tpr64")?;
    let pipe_csr = lib.pipe("disc_binary_csr_tpr64")?;
    let pipe_csr_ser = lib.pipe("disc_csr_serial_one_thread")?;
    let pipe_wh = lib.pipe("disc_walsh_hadamard_x")?;

    let (g64, t64) = tpr64_grid(rows);
    let (g256, t256) = tg256_grid(rows);
    let (gser, tser) = ((rows, 1, 1), (256, 1, 1));

    if !skip_f32 {
        if let Some(wp) = organ.get("W").and_then(|v| v.as_str()) {
            let w = read_bytes(wp)?;
            let w_buf = lib.buf(&w);
            let yref = read_f32(&format!(
                "{}/y_ref_dense.f32",
                Path::new(wp).parent().unwrap().display()
            ))
            .ok();
            let (med, reps, single) = lib.time_reps(|cmd| {
                let enc = cmd.new_compute_command_encoder();
                dispatch(&enc, &pipe_f32, g64, t64);
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&y_buf), 0);
                Lib::set_u32(&enc, 3, rows);
                Lib::set_u32(&enc, 4, cols);
                enc.end_encoding();
            })?;
            let got = read_out(&y_buf, rows as usize);
            if let Some(y) = yref.as_ref() {
                eprintln!(
                    "    f32 sample got[:4]={:?} y[:4]={:?} max_abs={:.4}",
                    &got[..4],
                    &y[..4],
                    max_abs(&got, y)
                );
            }
            results.push(variant(
                "f32_tpr64",
                "disc_f32_tpr64",
                w.len() as u64 + (cols as u64 * 4),
                med,
                &reps,
                yref.as_ref().map(|y| (got.as_slice(), y.as_slice())),
                "uncompressed occupancy-tile control",
            ));
        }
    }

    // production nibble q4
    if let Some(c) = codecs.get("prod_q4_nibble_g64") {
        let scales = lib.buf(&read_bytes(c["scales"].as_str().unwrap())?);
        let codes = lib.buf(&read_bytes(c["codes"].as_str().unwrap())?);
        let yhat = read_f32(&format!(
            "{}/prod_q4_nibble_g64.y_hat.f32",
            Path::new(c["scales"].as_str().unwrap())
                .parent()
                .unwrap()
                .display()
        ))?;
        let bytes = c["payload_bytes"].as_u64().unwrap();
        let (med, reps, single) = lib.time_reps(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            dispatch(&enc, &pipe_nibble, g64, t64);
            enc.set_buffer(0, Some(&codes), 0);
            enc.set_buffer(1, Some(&scales), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&y_buf), 0);
            Lib::set_u32(&enc, 4, rows);
            Lib::set_u32(&enc, 5, cols);
            enc.end_encoding();
        })?;
        let got = read_out(&y_buf, rows as usize);
        results.push(variant(
            "prod_q4_nibble_g64",
            "disc_q4_nibble_tpr64",
            bytes,
            med,
            &reps,
            Some((&got, &yhat)),
            "production Qwen3.8 q4 layout, occupancy tile",
        ));
    }

    let run_uniform = |lib: &Lib,
                       results: &mut Vec<Value>,
                       key: &str,
                       pipe: &ComputePipelineState,
                       grid: (u32, u32, u32),
                       tg: (u32, u32, u32),
                       kname: &str,
                       note: &str|
     -> Result<(), Box<dyn Error>> {
        let Some(c) = codecs.get(key) else {
            return Ok(());
        };
        let scales = lib.buf(&read_bytes(c["scales"].as_str().unwrap())?);
        let codes = lib.buf(&read_bytes(c["codes"].as_str().unwrap())?);
        let parent = Path::new(c["scales"].as_str().unwrap())
            .parent()
            .unwrap()
            .display()
            .to_string();
        let yhat = read_f32(&format!("{parent}/{key}.y_hat.f32"))?;
        let bits = c["bits"].as_u64().unwrap() as u32;
        let bound = c["bound"].as_u64().unwrap() as u32;
        let gs = c["group_size"].as_u64().unwrap() as u32;
        let bytes = c["payload_bytes"].as_u64().unwrap();
        let (med, reps, single) = lib.time_reps(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            dispatch(&enc, pipe, grid, tg);
            enc.set_buffer(0, Some(&codes), 0);
            enc.set_buffer(1, Some(&scales), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&y_buf), 0);
            Lib::set_u32(&enc, 4, rows);
            Lib::set_u32(&enc, 5, cols);
            Lib::set_u32(&enc, 6, gs);
            Lib::set_u32(&enc, 7, bits);
            Lib::set_u32(&enc, 8, bound);
            enc.end_encoding();
        })?;
        let got = read_out(&y_buf, rows as usize);
        results.push(variant(
            &format!("{key}/{kname}"),
            kname,
            bytes,
            med,
            &reps,
            Some((&got, &yhat)),
            note,
        ));
        Ok(())
    };

    for key in ["uniform_q4_g64", "uniform_q3_g64", "uniform_q2_g64"] {
        run_uniform(
            lib,
            &mut results,
            key,
            &pipe_uni,
            g64,
            t64,
            "disc_uniform_bits_tpr64",
            "gravity bitstream, wide extract, occupancy tile",
        )?;
        run_uniform(
            lib,
            &mut results,
            key,
            &pipe_uni256,
            g256,
            t256,
            "disc_uniform_bits_tg256",
            "Q80-won 256 threads/row",
        )?;
    }
    // serial artifact on q4 only (expensive, one organ is enough — still run both, it's ~ms)
    run_uniform(
        lib,
        &mut results,
        "uniform_q4_g64",
        &pipe_uni_ser,
        gser,
        tser,
        "disc_uniform_bits_serial",
        "DISPROVEN PATH: 1 thread/row serial extract",
    )?;

    // binary
    if let Some(c) = codecs.get("binary_g128") {
        let scales = lib.buf(&read_bytes(c["scales"].as_str().unwrap())?);
        let signs = lib.buf(&read_bytes(c["signs"].as_str().unwrap())?);
        let parent = Path::new(c["scales"].as_str().unwrap())
            .parent()
            .unwrap()
            .display()
            .to_string();
        let yhat = read_f32(&format!("{parent}/binary_g128.y_hat.f32"))?;
        let gs = c["group_size"].as_u64().unwrap() as u32;
        let bytes = c["payload_bytes"].as_u64().unwrap();
        for (pipe, grid, tg, kname, note) in [
            (
                &pipe_bin,
                g64,
                t64,
                "disc_binary_tpr64",
                "in-register sign*scale, occupancy tile",
            ),
            (
                &pipe_bin256,
                g256,
                t256,
                "disc_binary_tg256",
                "Q80-won 256 threads/row",
            ),
        ] {
            let (med, reps, single) = lib.time_reps(|cmd| {
                let enc = cmd.new_compute_command_encoder();
                dispatch(&enc, pipe, grid, tg);
                enc.set_buffer(0, Some(&signs), 0);
                enc.set_buffer(1, Some(&scales), 0);
                enc.set_buffer(2, Some(&x_buf), 0);
                enc.set_buffer(3, Some(&y_buf), 0);
                Lib::set_u32(&enc, 4, rows);
                Lib::set_u32(&enc, 5, cols);
                Lib::set_u32(&enc, 6, gs);
                enc.end_encoding();
            })?;
            let got = read_out(&y_buf, rows as usize);
            results.push(variant(
                &format!("binary_g128/{kname}"),
                kname,
                bytes,
                med,
                &reps,
                Some((&got, &yhat)),
                note,
            ));
        }
    }

    // ternary
    if let Some(c) = codecs.get("ternary_t0.7_g128") {
        let scales = lib.buf(&read_bytes(c["scales"].as_str().unwrap())?);
        let codes = lib.buf(&read_bytes(c["codes"].as_str().unwrap())?);
        let parent = Path::new(c["scales"].as_str().unwrap())
            .parent()
            .unwrap()
            .display()
            .to_string();
        let yhat = read_f32(&format!("{parent}/ternary_t0.7_g128.y_hat.f32"))?;
        let gs = c["group_size"].as_u64().unwrap() as u32;
        let bytes = c["payload_bytes"].as_u64().unwrap();
        let (med, reps, single) = lib.time_reps(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            dispatch(&enc, &pipe_ter, g64, t64);
            enc.set_buffer(0, Some(&codes), 0);
            enc.set_buffer(1, Some(&scales), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&y_buf), 0);
            Lib::set_u32(&enc, 4, rows);
            Lib::set_u32(&enc, 5, cols);
            Lib::set_u32(&enc, 6, gs);
            enc.end_encoding();
        })?;
        let got = read_out(&y_buf, rows as usize);
        results.push(variant(
            "ternary_t0.7_g128",
            "disc_ternary_tpr64",
            bytes,
            med,
            &reps,
            Some((&got, &yhat)),
            "2-bit in-register {0,+s,-s}",
        ));
    }

    // additive
    if let Some(c) = codecs.get("additive_q2q2_g64") {
        let bs = lib.buf(&read_bytes(c["base_scales"].as_str().unwrap())?);
        let rs = lib.buf(&read_bytes(c["residual_scales"].as_str().unwrap())?);
        let bc = lib.buf(&read_bytes(c["base_codes"].as_str().unwrap())?);
        let rc = lib.buf(&read_bytes(c["residual_codes"].as_str().unwrap())?);
        let parent = Path::new(c["base_scales"].as_str().unwrap())
            .parent()
            .unwrap()
            .display()
            .to_string();
        let yhat = read_f32(&format!("{parent}/additive_q2q2_g64.y_hat.f32"))?;
        let gs = c["group_size"].as_u64().unwrap() as u32;
        let bytes = c["payload_bytes"].as_u64().unwrap();
        let (med, reps, single) = lib.time_reps(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            dispatch(&enc, &pipe_add, g64, t64);
            enc.set_buffer(0, Some(&bc), 0);
            enc.set_buffer(1, Some(&bs), 0);
            enc.set_buffer(2, Some(&rc), 0);
            enc.set_buffer(3, Some(&rs), 0);
            enc.set_buffer(4, Some(&x_buf), 0);
            enc.set_buffer(5, Some(&y_buf), 0);
            Lib::set_u32(&enc, 6, rows);
            Lib::set_u32(&enc, 7, cols);
            Lib::set_u32(&enc, 8, gs);
            enc.end_encoding();
        })?;
        let got = read_out(&y_buf, rows as usize);
        results.push(variant(
            "additive_q2q2_g64",
            "disc_additive_tpr64",
            bytes,
            med,
            &reps,
            Some((&got, &yhat)),
            "two in-register q2 lookups",
        ));
    }

    // hadamard: WH(x) + q2 matvec in one CB
    if let Some(c) = codecs.get("hadamard_q2_g128") {
        let scales = lib.buf(&read_bytes(c["scales"].as_str().unwrap())?);
        let codes = lib.buf(&read_bytes(c["codes"].as_str().unwrap())?);
        let parent = Path::new(c["scales"].as_str().unwrap())
            .parent()
            .unwrap()
            .display()
            .to_string();
        let yhat = read_f32(&format!("{parent}/hadamard_q2_g128.y_hat.f32"))?;
        let xh_buf = lib.buf_len(cols as usize * 4);
        let bits = 2u32;
        let bound = 1u32;
        let gs = 128u32;
        let bytes = c["payload_bytes"].as_u64().unwrap();
        let ng = cols / 128;
        let (med, reps, single) = lib.time_reps(|cmd| {
            {
                let enc = cmd.new_compute_command_encoder();
                dispatch(&enc, &pipe_wh, (ng, 1, 1), (32, 1, 1));
                enc.set_buffer(0, Some(&x_buf), 0);
                enc.set_buffer(1, Some(&xh_buf), 0);
                Lib::set_u32(&enc, 2, cols);
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                dispatch(&enc, &pipe_uni, g64, t64);
                enc.set_buffer(0, Some(&codes), 0);
                enc.set_buffer(1, Some(&scales), 0);
                enc.set_buffer(2, Some(&xh_buf), 0);
                enc.set_buffer(3, Some(&y_buf), 0);
                Lib::set_u32(&enc, 4, rows);
                Lib::set_u32(&enc, 5, cols);
                Lib::set_u32(&enc, 6, gs);
                Lib::set_u32(&enc, 7, bits);
                Lib::set_u32(&enc, 8, bound);
                enc.end_encoding();
            }
        })?;
        let got = read_out(&y_buf, rows as usize);
        results.push(variant(
            "hadamard_q2_g128",
            "disc_walsh_hadamard_x+disc_uniform_bits_tpr64",
            bytes,
            med,
            &reps,
            Some((&got, &yhat)),
            "per-token WH(x) + in-register q2; WH is O(cols log 128)",
        ));
    }

    // rice: CSR consume (Q80-won) + serial diagnostic
    if let Some(c) = codecs.get("rice_q1_rms_2pct") {
        let scales = lib.buf(&read_bytes(c["scales"].as_str().unwrap())?);
        let signs = lib.buf(&read_bytes(c["signs"].as_str().unwrap())?);
        let csr_cols = lib.buf(&read_bytes(c["csr_cols"].as_str().unwrap())?);
        let csr_rp = lib.buf(&read_bytes(c["csr_row_ptr"].as_str().unwrap())?);
        let csr_sg = lib.buf(&read_bytes(c["csr_signs"].as_str().unwrap())?);
        let csr_sc = lib.buf(&read_bytes(c["csr_scale"].as_str().unwrap())?);
        let parent = Path::new(c["scales"].as_str().unwrap())
            .parent()
            .unwrap()
            .display()
            .to_string();
        let yhat = read_f32(&format!("{parent}/rice_q1_rms_2pct.y_hat.f32"))?;
        let gs = c["group_size"].as_u64().unwrap() as u32;
        let store_b = c["payload_bytes"].as_u64().unwrap();
        let traffic = c["csr_traffic_bytes"].as_u64().unwrap();
        let nnz = c["outlier_count"].as_u64().unwrap() as u32;
        let (med, reps, single) = lib.time_reps(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            dispatch(&enc, &pipe_csr, g64, t64);
            enc.set_buffer(0, Some(&signs), 0);
            enc.set_buffer(1, Some(&scales), 0);
            enc.set_buffer(2, Some(&csr_cols), 0);
            enc.set_buffer(3, Some(&csr_rp), 0);
            enc.set_buffer(4, Some(&csr_sg), 0);
            enc.set_buffer(5, Some(&csr_sc), 0);
            enc.set_buffer(6, Some(&x_buf), 0);
            enc.set_buffer(7, Some(&y_buf), 0);
            Lib::set_u32(&enc, 8, rows);
            Lib::set_u32(&enc, 9, cols);
            Lib::set_u32(&enc, 10, gs);
            enc.end_encoding();
        })?;
        let got = read_out(&y_buf, rows as usize);
        let mut v = variant(
            "rice_q1_rms_2pct/csr_inregister",
            "disc_binary_csr_tpr64",
            traffic,
            med,
            &reps,
            Some((&got, &yhat)),
            "Q80-won: bind-expanded CSR consumed in-register with binary",
        );
        v["storage_bytes"] = json!(store_b);
        v["storage_bpw"] = json!(8.0 * store_b as f64 / (rows as f64 * cols as f64));
        results.push(v);

        // serial one-thread residual walk (artifact)
        zero_out(&y_buf, 1);
        let (med, reps, single) = lib.time_reps(|cmd| {
            let enc = cmd.new_compute_command_encoder();
            dispatch(&enc, &pipe_csr_ser, (1, 1, 1), (1, 1, 1));
            enc.set_buffer(0, Some(&csr_cols), 0);
            enc.set_buffer(1, Some(&csr_sg), 0);
            enc.set_buffer(2, Some(&csr_sc), 0);
            enc.set_buffer(3, Some(&x_buf), 0);
            enc.set_buffer(4, Some(&y_buf), 0);
            Lib::set_u32(&enc, 5, nnz);
            Lib::set_u32(&enc, 6, cols);
            enc.end_encoding();
        })?;
        results.push(variant(
            "rice_q1_rms_2pct/serial_one_thread",
            "disc_csr_serial_one_thread",
            store_b,
            med,
            &reps,
            None,
            "ARTIFACT PATH: one thread walks every outlier. Not used for ranking.",
        ));
    }

    // two-stage
    if let Some(c) = codecs.get("hgravs01_r160_q3") {
        let lsc = lib.buf(&read_bytes(c["left_scales"].as_str().unwrap())?);
        let lco = lib.buf(&read_bytes(c["left_codes"].as_str().unwrap())?);
        let rsc = lib.buf(&read_bytes(c["right_scales"].as_str().unwrap())?);
        let rco = lib.buf(&read_bytes(c["right_codes"].as_str().unwrap())?);
        let parent = Path::new(c["left_scales"].as_str().unwrap())
            .parent()
            .unwrap()
            .display()
            .to_string();
        let yhat = read_f32(&format!("{parent}/hgravs01_r160_q3.y_hat.f32"))?;
        let rank = c["rank"].as_u64().unwrap() as u32;
        let bits = 3u32;
        let bound = 3u32;
        let gs = 64u32;
        let bytes = c["payload_bytes"].as_u64().unwrap();
        let mid = lib.buf_len(rank as usize * 4);
        let (rg, rt) = tpr64_grid(rank);
        let (lg, lt) = tpr64_grid(rows);
        let (med, reps, single) = lib.time_reps(|cmd| {
            {
                let enc = cmd.new_compute_command_encoder();
                dispatch(&enc, &pipe_uni, rg, rt);
                enc.set_buffer(0, Some(&rco), 0);
                enc.set_buffer(1, Some(&rsc), 0);
                enc.set_buffer(2, Some(&x_buf), 0);
                enc.set_buffer(3, Some(&mid), 0);
                Lib::set_u32(&enc, 4, rank);
                Lib::set_u32(&enc, 5, cols);
                Lib::set_u32(&enc, 6, gs);
                Lib::set_u32(&enc, 7, bits);
                Lib::set_u32(&enc, 8, bound);
                enc.end_encoding();
            }
            {
                let enc = cmd.new_compute_command_encoder();
                dispatch(&enc, &pipe_uni, lg, lt);
                enc.set_buffer(0, Some(&lco), 0);
                enc.set_buffer(1, Some(&lsc), 0);
                enc.set_buffer(2, Some(&mid), 0);
                enc.set_buffer(3, Some(&y_buf), 0);
                Lib::set_u32(&enc, 4, rows);
                Lib::set_u32(&enc, 5, rank);
                Lib::set_u32(&enc, 6, gs);
                Lib::set_u32(&enc, 7, bits);
                Lib::set_u32(&enc, 8, bound);
                enc.end_encoding();
            }
        })?;
        let got = read_out(&y_buf, rows as usize);
        results.push(variant(
            "hgravs01_r160_q3",
            "L@(R@x) two disc_uniform_bits_tpr64",
            bytes,
            med,
            &reps,
            Some((&got, &yhat)),
            "two-stage algebra, never reconstructs W. Rank 160 on real X.",
        ));
    }

    Ok(json!({
        "name": name,
        "rows": rows,
        "cols": cols,
        "variants": results,
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut root = PathBuf::from("scratch/qwen38_recon");
    let mut out = PathBuf::from("receipts/ascent-2026-08-16/QWEN38_RECON_MEASURED.json");
    let mut args = std::env::args().skip(1);
    while let Some(f) = args.next() {
        match f.as_str() {
            "--root" => root = PathBuf::from(args.next().ok_or("missing --root")?),
            "--out" => out = PathBuf::from(args.next().ok_or("missing --out")?),
            other => return Err(format!("unsupported {other}").into()),
        }
    }
    let man: Value = serde_json::from_str(&fs::read_to_string(root.join("manifest.json"))?)?;
    let lib = Lib::new()?;
    eprintln!("device: {}", lib.device.name());

    // stream control
    let pipe_ctl = lib.pipe("disc_stream_control")?;
    let ctl_data = vec![0u8; CONTROL_BYTES as usize];
    let ctl_buf = lib.buf(&ctl_data);
    let ctl_out = lib.buf_len((CONTROL_TGS * CONTROL_TPTG * 4) as usize);
    let (ctl_med, ctl_reps) = lib.time_reps(|cmd| {
        let enc = cmd.new_compute_command_encoder();
        dispatch(
            &enc,
            &pipe_ctl,
            (CONTROL_TGS * CONTROL_TPTG, 1, 1),
            (CONTROL_TPTG, 1, 1),
        );
        enc.set_buffer(0, Some(&ctl_buf), 0);
        enc.set_buffer(1, Some(&ctl_out), 0);
        Lib::set_u32(&enc, 2, CONTROL_BYTES as u32);
        Lib::set_u32(&enc, 3, CONTROL_ITERS);
        enc.end_encoding();
    })?;
    let ctl_traffic = CONTROL_BYTES as u64 * CONTROL_ITERS as u64;
    let ctl_gbps = (ctl_traffic as f64) / (ctl_med as f64);

    let mut organs = Vec::new();
    for key in ["gate", "down"] {
        eprintln!("organ {key} …");
        organs.push(run_organ(&lib, &man["organs"][key], false)?);
    }

    let receipt = json!({
        "schema": "hawking.special_unit.qwen38_recon_measured.v1",
        "date": "2026-08-16",
        "lane": "qwen38-descent-rescreen",
        "device_name": lib.device.name(),
        "gpu_time_authority": "MTLCommandBuffer.GPUEndTime-GPUStartTime after wait",
        "launch_primary": "64 threads/row, TG 128, 2 rows/TG (production Qwen3.8 q4 winner)",
        "honest_ceiling_gbps": HONEST_CEILING_GBPS,
        "qwen38_achieved_gbps": QWEN38_ACHIEVED_GBPS,
        "activation": man["activation"],
        "control": {
            "kernel": "disc_stream_control",
            "median_gpu_ns": ctl_med,
            "gpu_ns": ctl_reps,
            "gbps": ctl_gbps,
            "traffic_bytes": ctl_traffic,
            "threadgroups": CONTROL_TGS,
            "threads_per_threadgroup": CONTROL_TPTG,
        },
        "organs": organs,
    });
    if let Some(p) = out.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(&out, serde_json::to_string_pretty(&receipt)?)?;
    eprintln!("wrote {}", out.display());
    Ok(())
}
