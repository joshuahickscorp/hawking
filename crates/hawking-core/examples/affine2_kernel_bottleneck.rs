//! Affine2 g64 NON-LOAD critical-path levers vs the tpr64 incumbent.
//!
//! Isolated GEMV (no 27B): no-op control, deliberately-bad control, and four
//! non-load levers (tgsb / pipe / splitk4 / accfuse). Production decode on
//! NOETIC_PARENT_A is optional (`--artifact-root`) and never mutates the parent.
//! qmvfast / wide64 / tgx are not in this harness (N018: they lost).
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core --example affine2_kernel_bottleneck
//! ./tools/gpu_lane_lock.sh n024-bottleneck \
//!   workspace/ops/build/rust/release-fast/examples/affine2_kernel_bottleneck \
//!   --isolated --reps 7 --out receipts/headless/_KERNEL_BOTTLENECK_isolated.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn usage() -> &'static str {
    "usage: affine2_kernel_bottleneck [--isolated] [--artifact-root DIR] \
        [--tokenizer PATH] [--reps N] [--warmup N] [--max-new-tokens N] \
        [--max-seq-len N] [--prompt TEXT] [--raw-prompt] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("affine2_kernel_bottleneck: {message}");
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
        fail("--reps must be >= 7 (S017 §37)");
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
    fail("affine2_bandwidth_ascent requires macOS Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    let args = parse_args();
    let isolated = isolated::measure(args.warmup, args.reps).unwrap_or_else(|e| fail(e));
    let mut doc = json!({
        "schema": "hawking.headless.kernel_bottleneck.raw.v1",
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
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
    use serde_json::{json, Value};
    use std::time::Instant;

    const SHADER: &str = include_str!("../shaders/affine2_group32_matvec.metal");
    const PASS_TOL: f32 = 2e-2;
    const PARENT_ACTIVE_BYTES: u64 = 9_878_901_136;
    const BAR_GB_S: f64 = 775.0;
    const SPEC_GB_S: f64 = 819.0;

    struct Arm {
        id: &'static str,
        kernel: &'static str,
        role: &'static str,
        lever: Option<&'static str>,
        rows_per_tg: u32,
        tg: u64,
        bind_group_size: bool,
    }

    const ARMS: &[Arm] = &[
        Arm {
            id: "tpr64",
            kernel: "affine2_group64_matvec_geo_tpr64_tg128",
            role: "no_op_control",
            lever: None,
            rows_per_tg: 2,
            tg: 128,
            bind_group_size: false,
        },
        Arm {
            id: "runtime_div",
            kernel: "affine2_group64_matvec_geo_tpr64_tg128_runtime_div",
            role: "deliberately_bad_control",
            lever: None,
            rows_per_tg: 2,
            tg: 128,
            bind_group_size: true,
        },
        Arm {
            id: "tgsb",
            kernel: "affine2_group64_matvec_tgsb_tpr64_tg128",
            role: "lever",
            lever: Some("threadgroup-staged scale/bias once per TG; tpr64 occupancy"),
            rows_per_tg: 2,
            tg: 128,
            bind_group_size: false,
        },
        Arm {
            id: "pipe",
            kernel: "affine2_group64_matvec_pipe_tpr64_tg128",
            role: "lever",
            lever: Some("software-pipeline next unpack + vectorized x; tpr64 occupancy"),
            rows_per_tg: 2,
            tg: 128,
            bind_group_size: false,
        },
        Arm {
            id: "splitk4",
            kernel: "affine2_group64_matvec_splitk4_tg256",
            role: "lever",
            lever: Some("4-way split-K, TG 256, 2 rows, stride 1024 (not tgx)"),
            rows_per_tg: 2,
            tg: 256,
            bind_group_size: false,
        },
        Arm {
            id: "accfuse",
            kernel: "affine2_group64_matvec_accfuse_tpr64_tg128",
            role: "lever",
            lever: Some("fuse scale/bias into acc: scale*sum(q x)+bias*sum(x)"),
            rows_per_tg: 2,
            tg: 128,
            bind_group_size: false,
        },
        Arm {
            id: "qmvfast_addr_probe",
            kernel: "affine2_group64_matvec_qmvfast_r8tg64_addr_probe",
            role: "load_only_ceiling",
            lever: Some("N018 load-only ceiling; skip (q*scale+bias)*x"),
            rows_per_tg: 8,
            tg: 64,
            bind_group_size: false,
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
    fn median_u64(mut v: Vec<u64>) -> Option<u64> {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    }
    fn gb_s(bytes: u64, ns: u64) -> Option<f64> {
        if ns == 0 {
            None
        } else {
            Some(bytes as f64 / ns as f64)
        }
    }

    fn packed_half_bufs(packed: &AffineFactorPacked) -> (Vec<u8>, Vec<u16>, Vec<u16>) {
        (
            packed.codes.clone(),
            packed.scales_f16.clone(),
            packed.biases_f16.clone(),
        )
    }

    fn dispatch(enc: &metal::ComputeCommandEncoderRef, arm: &Arm, rows: u32) {
        let groups = u64::from(rows.div_ceil(arm.rows_per_tg).max(1));
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(arm.tg, 1, 1));
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
        let mut pipes = Vec::new();
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
                "registers_note": "Metal pipeline state does not report register count on this toolchain",
            }));
            pipes.push(p);
        }

        // Parity on a small packed matrix (CPU oracle). Addr-probe is excluded.
        let parity_rows = 64usize;
        let parity_cols = 256usize;
        let values = deterministic_matrix(parity_rows, parity_cols, 41);
        let packed =
            pack_affine_factor_group(&values, parity_rows, parity_cols, AFFINE_GROUP_SIZE_64)
                .map_err(|e| e.to_string())?;
        let (codes, scales, biases) = packed_half_bufs(&packed);
        let input = fill_f32(parity_cols);
        let cpu_y = affine_factor_matvec_f32(&packed, &input).map_err(|e| e.to_string())?;
        let codes_buf = device.new_buffer_with_data(
            as_bytes_u8(&codes).as_ptr() as *const _,
            codes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scales_buf = device.new_buffer_with_data(
            as_bytes_u16(&scales).as_ptr() as *const _,
            as_bytes_u16(&scales).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let biases_buf = device.new_buffer_with_data(
            as_bytes_u16(&biases).as_ptr() as *const _,
            as_bytes_u16(&biases).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let input_buf = device.new_buffer_with_data(
            as_bytes_f32(&input).as_ptr() as *const _,
            as_bytes_f32(&input).len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let y_buf = device.new_buffer(
            (parity_rows * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut parity = Vec::new();
        for (arm, pipe) in ARMS.iter().zip(pipes.iter()) {
            if arm.role == "load_only_ceiling" {
                continue;
            }
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&codes_buf), 0);
            enc.set_buffer(1, Some(&scales_buf), 0);
            enc.set_buffer(2, Some(&biases_buf), 0);
            enc.set_buffer(3, Some(&input_buf), 0);
            enc.set_buffer(4, Some(&y_buf), 0);
            set_u32(enc, 5, parity_rows as u32);
            set_u32(enc, 6, parity_cols as u32);
            if arm.bind_group_size {
                set_u32(enc, 7, 64);
            }
            dispatch(&enc, arm, parity_rows as u32);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu_y = read_f32(&y_buf, parity_rows);
            let diff = max_abs(&gpu_y, &cpu_y);
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

        let shapes = [
            (17408u32, 5120u32, "mlp.gate_proj"),
            (5120u32, 17408u32, "mlp.down_proj"),
        ];
        let mut shape_json = Vec::new();
        for (rows, cols, label) in shapes {
            eprintln!("isolated GEMV {label} {rows}x{cols}");
            let n = (rows * cols) as usize;
            let codes = fill_u8(n / 4, 0xA264);
            let groups = (cols / 64) as usize;
            let scales = fill_u16(rows as usize * groups, 0x51);
            let biases = fill_u16(rows as usize * groups, 0xB1);
            let input = fill_f32(cols as usize);
            let payload = (codes.len() + scales.len() * 2 + biases.len() * 2) as u64;
            let x_bytes = (cols as u64) * 4;
            let y_bytes = (rows as u64) * 4;
            let stream_bytes = payload + x_bytes + y_bytes;
            let codes_buf = device.new_buffer_with_data(
                as_bytes_u8(&codes).as_ptr() as *const _,
                codes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let scales_buf = device.new_buffer_with_data(
                as_bytes_u16(&scales).as_ptr() as *const _,
                as_bytes_u16(&scales).len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let biases_buf = device.new_buffer_with_data(
                as_bytes_u16(&biases).as_ptr() as *const _,
                as_bytes_u16(&biases).len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let input_buf = device.new_buffer_with_data(
                as_bytes_f32(&input).as_ptr() as *const _,
                as_bytes_f32(&input).len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let y_buf = device.new_buffer(
                (rows as usize * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let mut arms_json = Vec::new();
            for (arm, pipe) in ARMS.iter().zip(pipes.iter()) {
                let run = |n: usize| -> Vec<u64> {
                    let mut gpu = Vec::new();
                    for _ in 0..n {
                        let cmd = queue.new_command_buffer();
                        let enc = cmd.new_compute_command_encoder();
                        enc.set_compute_pipeline_state(pipe);
                        enc.set_buffer(0, Some(&codes_buf), 0);
                        enc.set_buffer(1, Some(&scales_buf), 0);
                        enc.set_buffer(2, Some(&biases_buf), 0);
                        enc.set_buffer(3, Some(&input_buf), 0);
                        enc.set_buffer(4, Some(&y_buf), 0);
                        set_u32(enc, 5, rows);
                        set_u32(enc, 6, cols);
                        if arm.bind_group_size {
                            set_u32(enc, 7, 64);
                        }
                        dispatch(&enc, arm, rows);
                        enc.end_encoding();
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
                let gb_weight = med.and_then(|ns| gb_s(payload, ns));
                let gb_stream = med.and_then(|ns| gb_s(stream_bytes, ns));
                eprintln!(
                    "  {} median_gpu_ns={:?} weight_gb_s={:?}",
                    arm.id, med, gb_weight
                );
                arms_json.push(json!({
                    "id": arm.id,
                    "kernel": arm.kernel,
                    "role": arm.role,
                    "lever": arm.lever,
                    "rows_per_tg": arm.rows_per_tg,
                    "tg": arm.tg,
                    "warmup": warmup,
                    "reps": reps,
                    "gpu_ns_reps": gpu,
                    "gpu_ns_min": min,
                    "gpu_ns_median": med,
                    "gpu_ns_max": max,
                    "weight_payload_bytes": payload,
                    "stream_bytes": stream_bytes,
                    "weight_gb_s_median": gb_weight,
                    "stream_gb_s_median": gb_stream,
                    "dense_w_materialized": 0,
                }));
            }
            shape_json.push(json!({
                "label": label,
                "rows": rows,
                "cols": cols,
                "group_size": 64,
                "scale_dtype": "f16 (production byte mix)",
                "bias_dtype": "f16",
                "weight_payload_bytes": payload,
                "arms": arms_json,
            }));
        }

        Ok(json!({
            "compile_s": compile_s,
            "shader": "crates/hawking-core/shaders/affine2_group32_matvec.metal",
            "parity": parity,
            "occupancy": occupancy,
            "shapes": shape_json,
            "bar_gb_s": BAR_GB_S,
            "spec_peak_gb_s": SPEC_GB_S,
            "roof_gb_s": 778.8,
            "parent_active_bytes_note": PARENT_ACTIVE_BYTES,
            "dense_w_materialized": 0,
        }))
    }
}

#[cfg(target_os = "macos")]
mod production {
    use super::Args;
    use hawking_core::model::qwen38_hybrid_decode::{
        generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Affine2Geo,
        Qwen38GenerateResult, Qwen38HybridDecodeSession, Qwen38MlpFusion,
    };
    use serde_json::{json, Value};
    use std::path::Path;

    const PARENT_ACTIVE_BYTES: u64 = 9_878_901_136;
    const BAR_GB_S: f64 = 775.0;
    const SPEC_GB_S: f64 = 819.0;

    fn tok_s(result: &Qwen38GenerateResult) -> Option<f64> {
        if result.decode_steps == 0 || result.decode_wall_ns == 0 {
            return None;
        }
        Some(result.decode_steps as f64 / (result.decode_wall_ns as f64 / 1e9))
    }

    fn gb_s(bytes: u64, ns: u64) -> Option<f64> {
        if ns == 0 {
            None
        } else {
            Some(bytes as f64 / ns as f64)
        }
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
        Affine2Geo::RuntimeDiv,
        Affine2Geo::Tgsb,
        Affine2Geo::Pipe,
        Affine2Geo::SplitK4,
        Affine2Geo::AccFuse,
    ];

    pub fn measure(args: &Args, root: &Path, tokenizer: &Path) -> Result<Value, String> {
        eprintln!("production: load {}", root.display());
        let mut session =
            Qwen38HybridDecodeSession::open(root, args.max_seq_len).map_err(|e| e.to_string())?;
        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
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
                let disp = result.dispatches.last().copied();
                eprintln!(
                    "  {} rep {i}: tok/s={ts:?} gpu_ns={gpu:?} dispatches={disp:?} new={}",
                    geo.as_str(),
                    new_ids.len()
                );
                texts.push(text);
                ids.push(new_ids);
                tok_s_reps.push(ts);
                gpu_ns_reps.push(gpu);
                disp_reps.push(disp);
                fallbacks.push(result.fallbacks);
                wall_ns.push(result.decode_wall_ns);
            }
            let gpu_vals: Vec<u64> = gpu_ns_reps.iter().copied().flatten().collect();
            let med_gpu = median_u64(gpu_vals.clone());
            let gb = med_gpu.and_then(|ns| gb_s(PARENT_ACTIVE_BYTES, ns));
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
                Affine2Geo::RuntimeDiv => "deliberately_bad_control",
                Affine2Geo::Tgsb | Affine2Geo::Pipe | Affine2Geo::SplitK4 | Affine2Geo::AccFuse => {
                    "lever"
                }
                _ => "lever",
            };
            arms.push(json!({
                "id": geo.as_str(),
                "role": role,
                "affine2_geo": geo.as_str(),
                "mlp_fusion": "swiglu",
                "fuse_gqa_qkv": true,
                "fuse_dn_inproj": true,
                "reps": args.reps,
                "generated_text_verbatim": texts.first(),
                "new_token_ids": ids0,
                "new_token_ids_all_reps": ids,
                "token_ids_unchanged_vs_tpr64": token_ids_unchanged,
                "token_ids_stable_across_reps": ids_stable_across_reps,
                "tok_s_reps": tok_s_reps,
                "median_gpu_ns_per_token_reps": gpu_ns_reps,
                "gpu_ns_min": gpu_vals.iter().copied().min(),
                "gpu_ns_median": med_gpu,
                "gpu_ns_max": gpu_vals.iter().copied().max(),
                "achieved_gb_s_median": gb,
                "dispatches_last_step_reps": disp_reps,
                "decode_wall_ns_reps": wall_ns,
                "fallbacks_reps": fallbacks,
                "dense_w_materialized": 0,
                "active_bytes_per_token": PARENT_ACTIVE_BYTES,
                "bar_gb_s": BAR_GB_S,
                "spec_peak_gb_s": SPEC_GB_S,
            }));
        }
        Ok(json!({
            "artifact": root.display().to_string(),
            "fusion": "parent (mlp swiglu + gqa qkv + dn inproj)",
            "active_bytes_per_token": PARENT_ACTIVE_BYTES,
            "did_not_mutate_parent": true,
            "did_not_load_second_27b": true,
            "residency_env": std::env::var("HAWKING_QWEN_RESIDENCY").ok(),
            "concurrent_env": std::env::var("HAWKING_QWEN38_CONCURRENT").ok(),
            "arms": arms,
        }))
    }
}
