//! N030 — fused gate_up_swiglu NON-LOAD autopsy.
//!
//! Profiles packed decode, scale, accumulate, SwiGLU, and the 64 launches.
//! Attempts one reduction that is NOT load geometry (N024 ruled the tile out):
//! group-64 x-sums fused into RMSNorm, bias deferred out of the inner loop
//! (S022 §9 norm+projection). Ranked by COMPLETE_TOKEN_NS, not GB/s.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example mlp_gate_up
//! ./tools/gpu_lane_lock.sh n030-gateup \
//!   workspace/ops/build/rust/release-fast/examples/mlp_gate_up \
//!   --isolated --reps 7 --out receipts/headless/_MLP_GATE_UP_isolated.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: mlp_gate_up [--isolated] [--artifact-root DIR] [--tokenizer PATH] \
        [--reps N] [--warmup N] [--max-new-tokens N] [--max-seq-len N] \
        [--prompt TEXT] [--raw-prompt] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("mlp_gate_up: {message}");
    process::exit(2);
}

struct Args {
    isolated: bool,
    artifact_root: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    reps: usize,
    warmup: usize,
    max_new_tokens: usize,
    max_seq_len: usize,
    prompt: String,
    raw_prompt: bool,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut isolated = false;
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut reps = 7usize;
    let mut warmup = 3usize;
    let mut max_new_tokens = 16usize;
    let mut max_seq_len = 128usize;
    let mut prompt = concat!(
        "Explain, in ordinary prose and at length, how a compiler turns a ",
        "for-loop into basic blocks and then into machine code."
    )
    .to_owned();
    let mut raw_prompt = false;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--isolated" => isolated = true,
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
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
            "--max-new-tokens" => {
                max_new_tokens = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-new-tokens"));
            }
            "--max-seq-len" => {
                max_seq_len = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--max-seq-len"));
            }
            "--prompt" => prompt = args.next().unwrap_or_else(|| fail(usage())),
            "--raw-prompt" => raw_prompt = true,
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if reps < 7 {
        fail("--reps must be >= 7 (S017 §37 / S022 §36)");
    }
    Args {
        isolated,
        artifact_root,
        tokenizer,
        reps,
        warmup,
        max_new_tokens,
        max_seq_len,
        prompt,
        raw_prompt,
        out,
    }
}

fn write_json(path: &std::path::Path, body: &Value) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, serde_json::to_vec_pretty(body).expect("json")).unwrap_or_else(|e| fail(e));
    eprintln!("wrote {}", path.display());
}

#[cfg(not(target_os = "macos"))]
fn main() {
    fail("mlp_gate_up requires macOS Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    let isolated = isolated::measure(args.warmup, args.reps).unwrap_or_else(|e| fail(e));
    let mut doc = json!({
        "schema": "hawking.headless.mlp_gate_up.raw.v1",
        "isolated": isolated,
        "production": Value::Null,
    });
    if !args.isolated {
        if let (Some(root), Some(tok)) = (args.artifact_root.clone(), args.tokenizer.clone()) {
            match production::measure(&args, &root, &tok) {
                Ok(prod) => doc["production"] = prod,
                Err(e) => {
                    doc["production"] = json!({
                        "kind": "ABSENT",
                        "absent_reason": e,
                    });
                }
            }
        } else {
            doc["production"] = json!({
                "kind": "ABSENT",
                "absent_reason": "no --artifact-root/--tokenizer; isolated GEMV only",
            });
        }
    }
    if let Some(path) = &args.out {
        write_json(path, &doc);
    } else {
        println!("{}", serde_json::to_string_pretty(&doc).expect("json"));
    }
}

#[cfg(target_os = "macos")]
mod isolated {
    use hawking_core::model::qwen_complete_binary::{
        affine_factor_matvec_f32, deterministic_matrix, pack_affine_factor_group,
        AffineFactorPacked, AFFINE_GROUP_SIZE_64,
    };
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
    use serde_json::{json, Value};
    use std::time::Instant;

    const SHADER: &str = include_str!("../shaders/affine2_group32_matvec.metal");
    const PASS_TOL: f32 = 2e-2;
    const ROWS: u32 = 17408;
    const COLS: u32 = 5120;

    #[derive(Clone, Copy)]
    enum Kind {
        Fused,
        AccOnly,
        BiasPrep,
        Xsum,
    }

    struct Arm {
        id: &'static str,
        kernel: &'static str,
        role: &'static str,
        kind: Kind,
        expect_parity: bool,
        launches: u32,
    }

    const ARMS: &[Arm] = &[
        Arm {
            id: "tpr64",
            kernel: "affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
            role: "no_op_control",
            kind: Kind::Fused,
            expect_parity: true,
            launches: 1,
        },
        Arm {
            id: "acc_only",
            kernel: "affine2_group64_matvec_gate_up_geo_tpr64_tg128",
            role: "component_accumulate",
            kind: Kind::AccOnly,
            expect_parity: false,
            launches: 1,
        },
        Arm {
            id: "biasprep",
            kernel: "affine2_group64_matvec_gate_up_swiglu_biasprep_tpr64_tg128",
            role: "lever",
            kind: Kind::BiasPrep,
            expect_parity: true,
            launches: 1,
        },
        Arm {
            id: "dropbias",
            kernel: "affine2_group64_matvec_gate_up_swiglu_biasprep_drop_tpr64_tg128",
            role: "deliberately_bad_control",
            kind: Kind::BiasPrep,
            expect_parity: false,
            launches: 1,
        },
        Arm {
            id: "decode_probe",
            kernel: "affine2_group64_matvec_gate_up_swiglu_decode_probe_tpr64_tg128",
            role: "component_packed_decode",
            kind: Kind::Fused,
            expect_parity: false,
            launches: 1,
        },
        Arm {
            id: "addr_probe",
            kernel: "affine2_group64_matvec_gate_up_swiglu_addr_probe_tpr64_tg128",
            role: "component_load_only",
            kind: Kind::Fused,
            expect_parity: false,
            launches: 1,
        },
        Arm {
            id: "launch64",
            kernel: "affine2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
            role: "component_64_launches",
            kind: Kind::Fused,
            expect_parity: false,
            launches: 64,
        },
        Arm {
            id: "xsum64",
            kernel: "affine2_xsum64",
            role: "component_xsum",
            kind: Kind::Xsum,
            expect_parity: false,
            launches: 1,
        },
    ];

    fn as_bytes_u8(v: &[u8]) -> &[u8] {
        v
    }
    fn as_bytes_u16(v: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }
    }
    fn as_bytes_f32(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }
    fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        enc.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
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
    fn read_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
    }
    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }
    fn fill_u8(n: usize, seed: u64) -> Vec<u8> {
        (0..n)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed);
                (x >> 33) as u8
            })
            .collect()
    }
    fn fill_u16(n: usize, seed: u64) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let x = (i as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .wrapping_add(seed);
                0x2c00u16.wrapping_add((x as u16) & 0x03ff)
            })
            .collect()
    }
    fn fill_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i % 17) as f32 * 0.125 - 1.0).collect()
    }
    fn xsum64_cpu(x: &[f32]) -> Vec<f32> {
        x.chunks(64).map(|c| c.iter().copied().sum::<f32>()).collect()
    }
    fn silu(g: f32) -> f32 {
        g / (1.0 + (-g).exp())
    }
    fn median_u64(mut v: Vec<u64>) -> Option<u64> {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    }
    fn packed_half_bufs(packed: &AffineFactorPacked) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
        (
            packed.codes.clone(),
            packed.scales_f16.clone(),
            packed.biases_f16.clone(),
        )
    }
    fn dispatch_fused(enc: &metal::ComputeCommandEncoderRef, rows: u32) {
        let groups = u64::from(rows.div_ceil(2).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
    }

    fn bind_fused(
        enc: &metal::ComputeCommandEncoderRef,
        arm: &Arm,
        gate_codes: &metal::Buffer,
        gate_scales: &metal::Buffer,
        gate_biases: &metal::Buffer,
        up_codes: &metal::Buffer,
        up_scales: &metal::Buffer,
        up_biases: &metal::Buffer,
        input: &metal::Buffer,
        act: &metal::Buffer,
        gate_out: &metal::Buffer,
        up_out: &metal::Buffer,
        xsum: &metal::Buffer,
        rows: u32,
        cols: u32,
    ) {
        match arm.kind {
            Kind::Xsum => {
                enc.set_buffer(0, Some(input), 0);
                enc.set_buffer(1, Some(xsum), 0);
                set_u32(enc, 2, cols);
                enc.dispatch_threads(
                    MTLSize::new(u64::from((cols / 64).max(1)), 1, 1),
                    MTLSize::new(u64::from((cols / 64).min(64).max(1)), 1, 1),
                );
            }
            Kind::AccOnly => {
                enc.set_buffer(0, Some(gate_codes), 0);
                enc.set_buffer(1, Some(gate_scales), 0);
                enc.set_buffer(2, Some(gate_biases), 0);
                enc.set_buffer(3, Some(up_codes), 0);
                enc.set_buffer(4, Some(up_scales), 0);
                enc.set_buffer(5, Some(up_biases), 0);
                enc.set_buffer(6, Some(input), 0);
                enc.set_buffer(7, Some(gate_out), 0);
                enc.set_buffer(8, Some(up_out), 0);
                set_u32(enc, 9, rows);
                set_u32(enc, 10, cols);
                dispatch_fused(enc, rows);
            }
            Kind::BiasPrep => {
                enc.set_buffer(0, Some(gate_codes), 0);
                enc.set_buffer(1, Some(gate_scales), 0);
                enc.set_buffer(2, Some(gate_biases), 0);
                enc.set_buffer(3, Some(up_codes), 0);
                enc.set_buffer(4, Some(up_scales), 0);
                enc.set_buffer(5, Some(up_biases), 0);
                enc.set_buffer(6, Some(input), 0);
                enc.set_buffer(7, Some(act), 0);
                enc.set_buffer(8, Some(xsum), 0);
                set_u32(enc, 9, rows);
                set_u32(enc, 10, cols);
                dispatch_fused(enc, rows);
            }
            Kind::Fused => {
                enc.set_buffer(0, Some(gate_codes), 0);
                enc.set_buffer(1, Some(gate_scales), 0);
                enc.set_buffer(2, Some(gate_biases), 0);
                enc.set_buffer(3, Some(up_codes), 0);
                enc.set_buffer(4, Some(up_scales), 0);
                enc.set_buffer(5, Some(up_biases), 0);
                enc.set_buffer(6, Some(input), 0);
                enc.set_buffer(7, Some(act), 0);
                set_u32(enc, 8, rows);
                set_u32(enc, 9, cols);
                dispatch_fused(enc, rows);
            }
        }
    }

    pub fn measure(warmup: usize, reps: usize) -> Result<Value, String> {
        let device = Device::system_default().ok_or("no Metal-capable GPU")?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        eprintln!("isolated: compile affine2 standalone shader");
        let t0 = Instant::now();
        let lib = device
            .new_library_with_source(SHADER, &opts)
            .map_err(|e| format!("affine2 shader compile: {e}"))?;
        let compile_s = t0.elapsed().as_secs_f64();
        let mut pipes: Vec<ComputePipelineState> = Vec::new();
        let mut occupancy = Vec::new();
        for arm in ARMS {
            let f = lib
                .get_function(arm.kernel, None)
                .map_err(|e| format!("{}: {e}", arm.kernel))?;
            let p = device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {}: {e}", arm.kernel))?;
            occupancy.push(json!({
                "id": arm.id,
                "kernel": arm.kernel,
                "max_total_threads_per_threadgroup": p.max_total_threads_per_threadgroup(),
                "thread_execution_width": p.thread_execution_width(),
                "registers_exposed": false,
            }));
            pipes.push(p);
        }

        let parity_rows = 64usize;
        let parity_cols = 256usize;
        let gate_m = deterministic_matrix(parity_rows, parity_cols, 41);
        let up_m = deterministic_matrix(parity_rows, parity_cols, 97);
        let gate_p =
            pack_affine_factor_group(&gate_m, parity_rows, parity_cols, AFFINE_GROUP_SIZE_64)
                .map_err(|e| e.to_string())?;
        let up_p = pack_affine_factor_group(&up_m, parity_rows, parity_cols, AFFINE_GROUP_SIZE_64)
            .map_err(|e| e.to_string())?;
        let (gc, gs, gb) = packed_half_bufs(&gate_p);
        let (uc, us, ub) = packed_half_bufs(&up_p);
        let input = fill_f32(parity_cols);
        let xsum = xsum64_cpu(&input);
        let cpu_g = affine_factor_matvec_f32(&gate_p, &input).map_err(|e| e.to_string())?;
        let cpu_u = affine_factor_matvec_f32(&up_p, &input).map_err(|e| e.to_string())?;
        let cpu_act: Vec<f32> = cpu_g
            .iter()
            .zip(&cpu_u)
            .map(|(g, u)| silu(*g) * *u)
            .collect();
        let new_buf = |bytes: &[u8]| {
            device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let gc_buf = new_buf(as_bytes_u8(&gc));
        let gs_buf = new_buf(as_bytes_u16(&gs));
        let gb_buf = new_buf(as_bytes_u16(&gb));
        let uc_buf = new_buf(as_bytes_u8(&uc));
        let us_buf = new_buf(as_bytes_u16(&us));
        let ub_buf = new_buf(as_bytes_u16(&ub));
        let in_buf = new_buf(as_bytes_f32(&input));
        let xs_buf = new_buf(as_bytes_f32(&xsum));
        let act_buf = device.new_buffer(
            (parity_rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let go_buf = device.new_buffer(
            (parity_rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let uo_buf = device.new_buffer(
            (parity_rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut parity = Vec::new();
        for (arm, pipe) in ARMS.iter().zip(pipes.iter()) {
            if !arm.expect_parity {
                continue;
            }
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            bind_fused(
                enc, arm, &gc_buf, &gs_buf, &gb_buf, &uc_buf, &us_buf, &ub_buf, &in_buf,
                &act_buf, &go_buf, &uo_buf, &xs_buf, parity_rows as u32, parity_cols as u32,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_y = read_f32(&act_buf, parity_rows);
            let diff = max_abs(&gpu_y, &cpu_act);
            let ok = diff.is_finite() && diff < PASS_TOL;
            eprintln!(
                "isolated parity {}: max_abs={diff:.4e} {}",
                arm.id,
                if ok { "PASS" } else { "FAIL" }
            );
            parity.push(json!({
                "id": arm.id,
                "kernel": arm.kernel,
                "max_abs_diff": diff,
                "ok": ok,
                "tolerance": PASS_TOL,
                "dense_w_materialized": 0,
            }));
            if !ok {
                return Err(format!(
                    "parity failed for {}: max_abs_diff={diff} >= {PASS_TOL}",
                    arm.id
                ));
            }
        }
        // dropbias must NOT match the oracle
        {
            let arm = ARMS.iter().find(|a| a.id == "dropbias").unwrap();
            let pipe = pipes
                .iter()
                .zip(ARMS.iter())
                .find(|(_, a)| a.id == "dropbias")
                .map(|(p, _)| p)
                .unwrap();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            bind_fused(
                enc, arm, &gc_buf, &gs_buf, &gb_buf, &uc_buf, &us_buf, &ub_buf, &in_buf,
                &act_buf, &go_buf, &uo_buf, &xs_buf, parity_rows as u32, parity_cols as u32,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_y = read_f32(&act_buf, parity_rows);
            let diff = max_abs(&gpu_y, &cpu_act);
            let rejected = diff.is_finite() && diff >= PASS_TOL;
            eprintln!("isolated dropbias max_abs={diff:.4e} rejected={rejected}");
            parity.push(json!({
                "id": "dropbias",
                "kernel": arm.kernel,
                "max_abs_diff": diff,
                "ok": false,
                "must_fail": true,
                "rejected": rejected,
                "tolerance": PASS_TOL,
                "dense_w_materialized": 0,
            }));
            if !rejected {
                return Err(format!(
                    "dropbias control matched the oracle (max_abs={diff}); bad control is vacuous"
                ));
            }
        }

        eprintln!("isolated fused gate_up_swiglu {ROWS}x{COLS}");
        let n = (ROWS * COLS) as usize;
        let groups = (COLS / 64) as usize;
        let gc = fill_u8(n / 4, 0xA264);
        let uc = fill_u8(n / 4, 0xB173);
        let gs = fill_u16(ROWS as usize * groups, 0x51);
        let gb = fill_u16(ROWS as usize * groups, 0xB1);
        let us = fill_u16(ROWS as usize * groups, 0x62);
        let ub = fill_u16(ROWS as usize * groups, 0xC2);
        let input = fill_f32(COLS as usize);
        let xsum = xsum64_cpu(&input);
        let payload = (gc.len() + uc.len() + (gs.len() + gb.len() + us.len() + ub.len()) * 2) as u64;
        let gc_buf = new_buf(as_bytes_u8(&gc));
        let gs_buf = new_buf(as_bytes_u16(&gs));
        let gb_buf = new_buf(as_bytes_u16(&gb));
        let uc_buf = new_buf(as_bytes_u8(&uc));
        let us_buf = new_buf(as_bytes_u16(&us));
        let ub_buf = new_buf(as_bytes_u16(&ub));
        let in_buf = new_buf(as_bytes_f32(&input));
        let xs_buf = new_buf(as_bytes_f32(&xsum));
        let act_buf =
            device.new_buffer((ROWS as usize * 4) as u64, MTLResourceOptions::StorageModeShared);
        let go_buf =
            device.new_buffer((ROWS as usize * 4) as u64, MTLResourceOptions::StorageModeShared);
        let uo_buf =
            device.new_buffer((ROWS as usize * 4) as u64, MTLResourceOptions::StorageModeShared);

        let mut arms_json = Vec::new();
        for (arm, pipe) in ARMS.iter().zip(pipes.iter()) {
            let run = |n: usize| -> Vec<u64> {
                let mut gpu = Vec::new();
                for _ in 0..n {
                    let cmd = queue.new_command_buffer();
                    for _ in 0..arm.launches {
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(pipe);
                        bind_fused(
                            enc, arm, &gc_buf, &gs_buf, &gb_buf, &uc_buf, &us_buf, &ub_buf,
                            &in_buf, &act_buf, &go_buf, &uo_buf, &xs_buf, ROWS, COLS,
                        );
                        enc.end_encoding();
                    }
                    cmd.commit();
                    cmd.wait_until_completed();
                    if let Some(ns) = gpu_ns(cmd) {
                        gpu.push(ns);
                    }
                }
                gpu
            };
            let _ = run(warmup);
            let gpu = run(reps);
            let med = median_u64(gpu.clone());
            let min = gpu.iter().copied().min();
            let max = gpu.iter().copied().max();
            eprintln!("  {} median_gpu_ns={:?} launches={}", arm.id, med, arm.launches);
            arms_json.push(json!({
                "id": arm.id,
                "kernel": arm.kernel,
                "role": arm.role,
                "launches": arm.launches,
                "warmup": warmup,
                "reps": reps,
                "gpu_ns_reps": gpu,
                "gpu_ns_min": min,
                "gpu_ns_median": med,
                "gpu_ns_max": max,
                "weight_payload_bytes": payload,
                "dense_w_materialized": 0,
            }));
        }

        Ok(json!({
            "compile_s": compile_s,
            "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
            "shape": {"rows": ROWS, "cols": COLS, "label": "mlp.gate_up_swiglu"},
            "parity": parity,
            "occupancy": occupancy,
            "arms": arms_json,
            "did_not_load_second_27b": true,
            "dense_w_materialized": 0,
        }))
    }
}

#[cfg(target_os = "macos")]
mod production {
    use super::Args;
    use hawking_core::model::qwen38_hybrid_decode::{
        generate_greedy, load_qwen38_tokenizer, qwen38_fused_dispatches_per_token_full,
        render_qwen38_user_chat, Affine2Geo, Qwen38GenerateResult, Qwen38HybridDecodeSession,
        Qwen38MlpFusion,
    };
    use serde_json::{json, Value};
    use std::path::Path;

    const PARENT_ACTIVE_BYTES: u64 = 9_878_901_136;

    fn tok_s(result: &Qwen38GenerateResult) -> Option<f64> {
        if result.decode_steps == 0 || result.decode_wall_ns == 0 {
            return None;
        }
        Some(result.decode_steps as f64 / (result.decode_wall_ns as f64 / 1e9))
    }

    fn complete_token_ns(result: &Qwen38GenerateResult) -> Option<u64> {
        result.steady_decode_wall_ns_per_token()
    }

    fn median_u64(mut v: Vec<u64>) -> Option<u64> {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    }

    const GEOS: &[Affine2Geo] = &[
        Affine2Geo::Tpr64,
        Affine2Geo::BiasPrep,
        Affine2Geo::BiasPrepDrop,
    ];

    pub fn measure(args: &Args, root: &Path, tokenizer: &Path) -> Result<Value, String> {
        eprintln!("production: load {}", root.display());
        let mut session = Qwen38HybridDecodeSession::open(root, args.max_seq_len)
            .map_err(|e| e.to_string())?;
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
        session.set_fuse_add_rmsnorm(true, false);
        session.set_fuse_ba_delta(true, false);
        let expected = qwen38_fused_dispatches_per_token_full(
            Qwen38MlpFusion::GateUpSwiglu,
            true,
            true,
            true,
            true,
        );
        let tok = load_qwen38_tokenizer(tokenizer).map_err(|e| e.to_string())?;
        let prompt_ids = if args.raw_prompt {
            tok.encode(&args.prompt, true).map_err(|e| e.to_string())?
        } else {
            let rendered = render_qwen38_user_chat(&args.prompt);
            tok.encode(&rendered, true).map_err(|e| e.to_string())?
        };
        let mut arms = Vec::new();
        let mut baseline_ids: Option<Vec<u32>> = None;
        for geo in GEOS {
            session.apply_affine2_geo(*geo);
            eprintln!("production geo={} warmup", geo.as_str());
            session.reset();
            let _ = generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
                .map_err(|e| e.to_string())?;
            let mut texts = Vec::new();
            let mut ids = Vec::new();
            let mut tok_s_reps = Vec::new();
            let mut gpu_ns_reps = Vec::new();
            let mut complete_ns_reps = Vec::new();
            let mut disp_reps = Vec::new();
            let mut fallbacks = Vec::new();
            let mut wall_ns = Vec::new();
            for i in 0..args.reps {
                session.reset();
                let result = generate_greedy(&mut session, &prompt_ids, args.max_new_tokens)
                    .map_err(|e| e.to_string())?;
                let text = result.decode_new(&tok).map_err(|e| e.to_string())?;
                let new_ids = result.new_tokens().to_vec();
                let ts = tok_s(&result);
                let gpu = result.median_gpu_ns_per_token();
                let cns = complete_token_ns(&result);
                let disp = result.dispatches.last().copied();
                eprintln!(
                    "  {} rep {i}: complete_token_ns={cns:?} tok/s={ts:?} gpu_ns={gpu:?} dispatches={disp:?} new={}",
                    geo.as_str(),
                    new_ids.len()
                );
                texts.push(text);
                ids.push(new_ids);
                tok_s_reps.push(ts);
                gpu_ns_reps.push(gpu);
                complete_ns_reps.push(cns);
                disp_reps.push(disp);
                fallbacks.push(result.fallbacks);
                wall_ns.push(result.decode_wall_ns);
            }
            let gpu_vals: Vec<u64> = gpu_ns_reps.iter().copied().flatten().collect();
            let cns_vals: Vec<u64> = complete_ns_reps.iter().copied().flatten().collect();
            let med_gpu = median_u64(gpu_vals.clone());
            let med_cns = median_u64(cns_vals.clone());
            let ids0 = ids.first().cloned().unwrap_or_default();
            if geo.as_str() == "tpr64" {
                baseline_ids = Some(ids0.clone());
            }
            let token_ids_unchanged = match &baseline_ids {
                Some(base) => ids.iter().all(|v| v == base),
                None => false,
            };
            let ids_stable_across_reps = ids.windows(2).all(|w| w[0] == w[1]);
            let role = match geo {
                Affine2Geo::Tpr64 => "no_op_control",
                Affine2Geo::BiasPrepDrop => "deliberately_bad_control",
                Affine2Geo::BiasPrep => "lever",
                _ => "lever",
            };
            arms.push(json!({
                "id": geo.as_str(),
                "role": role,
                "affine2_geo": geo.as_str(),
                "mlp_fusion": "swiglu",
                "fuse_gqa_qkv": true,
                "fuse_dn_inproj": true,
                "fuse_add_rmsnorm": true,
                "fuse_ba_delta": true,
                "expected_dispatches": expected,
                "reps": args.reps,
                "generated_text_verbatim": texts.first(),
                "new_token_ids": ids0,
                "new_token_ids_all_reps": ids,
                "token_ids_unchanged_vs_tpr64": token_ids_unchanged,
                "token_ids_stable_across_reps": ids_stable_across_reps,
                "tok_s_reps": tok_s_reps,
                "median_gpu_ns_per_token_reps": gpu_ns_reps,
                "complete_token_ns_reps": complete_ns_reps,
                "complete_token_ns_min": cns_vals.iter().copied().min(),
                "complete_token_ns_median": med_cns,
                "complete_token_ns_max": cns_vals.iter().copied().max(),
                "gpu_ns_min": gpu_vals.iter().copied().min(),
                "gpu_ns_median": med_gpu,
                "gpu_ns_max": gpu_vals.iter().copied().max(),
                "dispatches_last_step_reps": disp_reps,
                "decode_wall_ns_reps": wall_ns,
                "fallbacks_reps": fallbacks,
                "dense_w_materialized": 0,
                "active_bytes_per_token": PARENT_ACTIVE_BYTES,
            }));
        }
        Ok(json!({
            "artifact": root.display().to_string(),
            "fusion": "580-dispatch graph (swiglu + gqa qkv + dn inproj + add_rmsnorm + ba_delta)",
            "expected_dispatches": expected,
            "active_bytes_per_token": PARENT_ACTIVE_BYTES,
            "did_not_mutate_parent": true,
            "did_not_load_second_27b": true,
            "ranking_metric": "COMPLETE_TOKEN_NS",
            "arms": arms,
        }))
    }
}
