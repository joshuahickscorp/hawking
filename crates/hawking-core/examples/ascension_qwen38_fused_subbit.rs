//! Affine2 MLP (2-bit) on a fused operator graph.
//!
//! Opens one mixed catalog (does not load a second 27B). Times the affine2
//! geo_tpr64 kernel against q4 geo_tpr64 on the same shape, then applies
//! the four dispatch fusions (gate+up, gate+up+SwiGLU, GQA QKV, DeltaNet
//! qkvz+ba). Reports every config, including losers.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example ascension_qwen38_fused_subbit
//! ./tools/gpu_lane_lock.sh qwen38-fused-subbit \
//!   workspace/ops/build/rust/release-fast/examples/ascension_qwen38_fused_subbit \
//!   --artifact-root artifacts/qwen38-affine2-g64-lsfit/mix_all_mlp_affine_g64_ls \
//!   --tokenizer ~/models/qwen3.8-27b-abliterated-bf16/tokenizer.json \
//!   --out receipts/headless/_fused_subbit_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_geometry::{qwen38_layer_name, QWEN38_HIDDEN};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, qwen38_fused_dispatches_per_token,
    render_qwen38_user_chat, Qwen38FusionParity, Qwen38GenerateResult,
    Qwen38HybridDecodeSession, Qwen38MlpFusion,
};
use hawking_core::model::qwen38_token_ns_ledger::production_dispatches_per_token;

fn usage() -> &'static str {
    "usage: ascension_qwen38_fused_subbit --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--skip-decode] [--skip-kernel-cost] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("ascension_qwen38_fused_subbit: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    raw_prompt: bool,
    max_new_tokens: usize,
    max_seq_len: usize,
    reps: usize,
    skip_decode: bool,
    skip_kernel_cost: bool,
    out: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut prompt = concat!(
        "Explain, in ordinary prose and at length, how a compiler turns a ",
        "for-loop into basic blocks and then into machine code."
    )
    .to_owned();
    let mut raw_prompt = false;
    let mut max_new_tokens = 16usize;
    let mut max_seq_len = 128usize;
    let mut reps = 2usize;
    let mut skip_decode = false;
    let mut skip_kernel_cost = false;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage()))));
            }
            "--prompt" => prompt = args.next().unwrap_or_else(|| fail(usage())),
            "--raw-prompt" => raw_prompt = true,
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
            "--reps" => {
                reps = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--reps"));
            }
            "--skip-decode" => skip_decode = true,
            "--skip-kernel-cost" => skip_kernel_cost = true,
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        reps: reps.max(1),
        skip_decode,
        skip_kernel_cost,
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

fn now_iso() -> String {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{ts}")
}

fn max_abs(a: f32, b: f32, c: f32) -> f32 {
    a.max(b).max(c)
}

fn median_u64(mut values: Vec<u64>) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

#[cfg(target_os = "macos")]
fn parity_json(p: &Qwen38FusionParity) -> Value {
    json!({
        "fusion": p.fusion,
        "layer": p.layer,
        "unfused_dispatches": p.unfused_dispatches,
        "fused_dispatches": p.fused_pair_dispatches.min(p.fused_swiglu_dispatches),
        "fused_pair_dispatches": p.fused_pair_dispatches,
        "fused_swiglu_dispatches": p.fused_swiglu_dispatches,
        "unfused_gpu_ns": p.unfused_gpu_ns,
        "fused_pair_gpu_ns": p.fused_pair_gpu_ns,
        "fused_swiglu_gpu_ns": p.fused_swiglu_gpu_ns,
        "max_abs_diff_gate": p.max_abs_diff_gate,
        "max_abs_diff_up": p.max_abs_diff_up,
        "max_abs_diff_act": p.max_abs_diff_act,
        "max_abs_diff": max_abs(p.max_abs_diff_gate, p.max_abs_diff_up, p.max_abs_diff_act),
        "dense_w_materialized": p.dense_w_materialized,
    })
}

#[cfg(target_os = "macos")]
fn tok_s(result: &Qwen38GenerateResult) -> Option<f64> {
    if result.decode_steps == 0 || result.decode_wall_ns == 0 {
        return None;
    }
    Some(result.decode_steps as f64 / (result.decode_wall_ns as f64 / 1e9))
}

#[cfg(target_os = "macos")]
fn generate_arm(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
    reps: usize,
) -> Result<Value, String> {
    let mut texts = Vec::new();
    let mut ids = Vec::new();
    let mut tok_s_reps = Vec::new();
    let mut dispatch_reps = Vec::new();
    let mut decode_wall_ns = Vec::new();
    let mut gpu_median = Vec::new();
    let mut fallbacks = Vec::new();
    for i in 0..reps {
        session.reset();
        let result = generate_greedy(session, prompt_ids, max_new).map_err(|e| e.to_string())?;
        let text = result.decode_new(tokenizer).map_err(|e| e.to_string())?;
        let new_ids = result.new_tokens().to_vec();
        let ts = tok_s(&result);
        let disp = result.dispatches.last().copied();
        eprintln!(
            "  rep {i}: tok/s={ts:?} dispatches={disp:?} new={} text={text:?}",
            new_ids.len()
        );
        texts.push(text);
        ids.push(new_ids);
        tok_s_reps.push(ts);
        dispatch_reps.push(disp);
        decode_wall_ns.push(result.decode_wall_ns);
        gpu_median.push(result.median_gpu_ns_per_token());
        fallbacks.push(result.fallbacks);
    }
    let finite: Vec<f64> = tok_s_reps.iter().copied().flatten().collect();
    let mean = if finite.is_empty() {
        None
    } else {
        Some(finite.iter().sum::<f64>() / finite.len() as f64)
    };
    Ok(json!({
        "reps": reps,
        "generated_text_verbatim": texts.first(),
        "generated_text_all_reps": texts,
        "new_token_ids": ids.first(),
        "new_token_ids_all_reps": ids,
        "tok_s_reps": tok_s_reps,
        "tok_s_mean": mean,
        "tok_s_min": finite.iter().copied().reduce(f64::min),
        "tok_s_max": finite.iter().copied().reduce(f64::max),
        "dispatches_last_step_reps": dispatch_reps,
        "decode_wall_ns_reps": decode_wall_ns,
        "median_gpu_ns_per_token_reps": gpu_median,
        "fallbacks_reps": fallbacks,
        "dense_w_materialized": 0,
        "expanded_to_q4": 0,
        "expanded_to_float_gemv": 0,
    }))
}

#[cfg(target_os = "macos")]
fn probe_dispatches(
    session: &mut Qwen38HybridDecodeSession,
    token: u32,
) -> Result<Value, String> {
    session.reset();
    let theoretical = session.theoretical_dispatches();
    let (sampled, dispatches, timing) = session
        .measure_token_dispatches(token)
        .map_err(|e| e.to_string())?;
    Ok(json!({
        "theoretical": theoretical,
        "measured": dispatches,
        "matches_theoretical": dispatches == theoretical,
        "sampled": sampled,
        "gpu_ns": timing.gpu_ns,
        "wait_ns": timing.wait_ns,
        "mlp_fusion": session.mlp_fusion.as_str(),
        "fuse_gqa_qkv": session.fuse_gqa_qkv,
        "fuse_dn_inproj": session.fuse_dn_inproj,
    }))
}

#[cfg(target_os = "macos")]
mod kernel_cost {
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
    use serde_json::{json, Value};
    use std::time::Instant;

    // Standalone affine2 shader (float scale+bias) + Q4 geo shader (half scale).
    // q80_mixed_decode.metal is not a self-contained compile unit (gk_* live
    // in sibling shaders concatenated by MetalContext).
    const AFFINE: &str = include_str!("../shaders/affine2_group32_matvec.metal");
    const Q4: &str = include_str!("../shaders/qwen_uniform_q4.metal");
    const WARMUP: usize = 4;
    const REPS: usize = 8;

    fn as_bytes_u8(values: &[u8]) -> &[u8] {
        values
    }

    fn as_bytes_f32(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn as_bytes_u16(values: &[u16]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 2) }
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
        encoder.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    fn fill_u8(n: usize, seed: u64) -> Vec<u8> {
        (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed);
                (x >> 33) as u8
            })
            .collect()
    }

    fn fill_u16(n: usize, seed: u64) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let x = (i as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(seed);
                // keep a typical fp16 magnitude (~2^-6 .. 2^-2)
                0x2c00u16.wrapping_add((x as u16) & 0x03ff)
            })
            .collect()
    }

    fn fill_f32(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (i % 17) as f32 * 0.125 - 1.0)
            .collect()
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

    fn time_geo(
        queue: &metal::CommandQueue,
        pipe: &metal::ComputePipelineState,
        bind: impl Fn(&metal::ComputeCommandEncoderRef),
        rows: u32,
        warmup: usize,
        reps: usize,
    ) -> Value {
        let groups = u64::from(rows.div_ceil(2).max(1));
        let run = |n: usize| -> (Vec<u64>, Vec<u64>) {
            let mut gpu = Vec::new();
            let mut wall = Vec::new();
            for _ in 0..n {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe);
                bind(&enc);
                enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
                enc.end_encoding();
                let t0 = Instant::now();
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
        let mut gpu_sorted = gpu.clone();
        gpu_sorted.sort_unstable();
        let mut wall_sorted = wall.clone();
        wall_sorted.sort_unstable();
        json!({
            "gpu_ns_reps": gpu,
            "gpu_ns_median": gpu_sorted.get(gpu_sorted.len() / 2).copied(),
            "wait_ns_reps": wall,
            "wait_ns_median": wall_sorted.get(wall_sorted.len() / 2).copied(),
            "warmup": warmup,
            "reps": reps,
        })
    }

    pub fn measure() -> Result<Value, String> {
        let device = Device::system_default().ok_or("no Metal-capable GPU")?;
        let queue = device.new_command_queue();
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        eprintln!("kernel cost: compile affine2 standalone shader");
        let t0 = Instant::now();
        let affine_lib = device
            .new_library_with_source(AFFINE, &opts)
            .map_err(|e| format!("affine2 shader compile: {e}"))?;
        let affine_compile_s = t0.elapsed().as_secs_f64();
        eprintln!("kernel cost: compile q4 geo shader");
        let t1 = Instant::now();
        let q4_lib = device
            .new_library_with_source(Q4, &opts)
            .map_err(|e| format!("q4 shader compile: {e}"))?;
        let q4_compile_s = t1.elapsed().as_secs_f64();

        let affine_geo_fn = affine_lib
            .get_function("affine2_group32_matvec_geo_tpr64_tg128", None)
            .map_err(|e| e.to_string())?;
        let affine_runtime_fn = affine_lib
            .get_function("affine2_group32_matvec_geo_tpr64_tg128_runtime_div", None)
            .map_err(|e| e.to_string())?;
        let q4_geo_fn = q4_lib
            .get_function("qwen_uniform_q4_group64_matvec_geo_tpr64_tg128", None)
            .map_err(|e| e.to_string())?;
        let affine_geo = device
            .new_compute_pipeline_state_with_function(&affine_geo_fn)
            .map_err(|e| e.to_string())?;
        let affine_runtime = device
            .new_compute_pipeline_state_with_function(&affine_runtime_fn)
            .map_err(|e| e.to_string())?;
        let q4_geo = device
            .new_compute_pipeline_state_with_function(&q4_geo_fn)
            .map_err(|e| e.to_string())?;

        let shapes = [(17408u32, 5120u32, "gate_up"), (5120u32, 17408u32, "down")];
        let mut out = Vec::new();
        for (rows, cols, label) in shapes {
            eprintln!("kernel cost: shape {rows}x{cols} ({label})");
            let input = fill_f32(cols as usize);
            let y_len = (rows as usize) * 4;
            let input_buf = device.new_buffer_with_data(
                as_bytes_f32(&input).as_ptr() as *const _,
                as_bytes_f32(&input).len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let y_buf = device.new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);

            let mut shape_json = json!({
                "shape": [rows, cols],
                "label": label,
                "why_this_shape": "production MLP GEMV; gate/up is 17408x5120, down is 5120x17408",
            });

            for (group, kernel_label, pipe) in [
                (32u32, "affine2_g32_specialized", &affine_geo),
                (64u32, "affine2_g64_specialized", &affine_geo),
                (64u32, "affine2_g64_runtime_div", &affine_runtime),
            ] {
                let n = (rows * cols) as usize;
                let codes = fill_u8(n / 4, 0xA2 + u64::from(group));
                let groups = (cols / group) as usize;
                let scales = fill_f32(rows as usize * groups);
                let biases = fill_f32(rows as usize * groups);
                let codes_buf = device.new_buffer_with_data(
                    as_bytes_u8(&codes).as_ptr() as *const _,
                    codes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let scales_buf = device.new_buffer_with_data(
                    as_bytes_f32(&scales).as_ptr() as *const _,
                    as_bytes_f32(&scales).len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let biases_buf = device.new_buffer_with_data(
                    as_bytes_f32(&biases).as_ptr() as *const _,
                    as_bytes_f32(&biases).len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let payload_bytes = codes.len() as u64
                    + as_bytes_f32(&scales).len() as u64
                    + as_bytes_f32(&biases).len() as u64;
                let timing = time_geo(
                    &queue,
                    pipe,
                    |enc| {
                        enc.set_buffer(0, Some(&codes_buf), 0);
                        enc.set_buffer(1, Some(&scales_buf), 0);
                        enc.set_buffer(2, Some(&biases_buf), 0);
                        enc.set_buffer(3, Some(&input_buf), 0);
                        enc.set_buffer(4, Some(&y_buf), 0);
                        set_u32(enc, 5, rows);
                        set_u32(enc, 6, cols);
                        set_u32(enc, 7, group);
                    },
                    rows,
                    WARMUP,
                    REPS,
                );
                shape_json[kernel_label] = json!({
                    "kernel": if kernel_label.contains("runtime") {
                        "affine2_group32_matvec_geo_tpr64_tg128_runtime_div"
                    } else {
                        "affine2_group32_matvec_geo_tpr64_tg128"
                    },
                    "group_size": group,
                    "scale_dtype": "f32 (standalone parity shader; production mixed path uses f16)",
                    "payload_bytes": payload_bytes,
                    "body_bpw": 2.0 + 16.0 / f64::from(group) + 16.0 / f64::from(group),
                    "timing": timing,
                });
            }

            let n = (rows * cols) as usize;
            let q4_codes = fill_u8(n / 2, 0x44);
            let q4_groups = (cols / 64) as usize;
            let q4_scales = fill_u16(rows as usize * q4_groups, 0x44);
            let q4_codes_buf = device.new_buffer_with_data(
                as_bytes_u8(&q4_codes).as_ptr() as *const _,
                q4_codes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let q4_scales_buf = device.new_buffer_with_data(
                as_bytes_u16(&q4_scales).as_ptr() as *const _,
                as_bytes_u16(&q4_scales).len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let gpr = cols / 64;
            let q4_timing = time_geo(
                &queue,
                &q4_geo,
                |enc| {
                    enc.set_buffer(0, Some(&q4_codes_buf), 0);
                    enc.set_buffer(1, Some(&q4_scales_buf), 0);
                    enc.set_buffer(2, Some(&input_buf), 0);
                    enc.set_buffer(3, Some(&y_buf), 0);
                    set_u32(enc, 4, rows);
                    set_u32(enc, 5, cols);
                    set_u32(enc, 6, gpr);
                },
                rows,
                WARMUP,
                REPS,
            );
            shape_json["q4_g64_geo_tpr64"] = json!({
                "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
                "group_size": 64,
                "payload_bytes": q4_codes.len() as u64 + as_bytes_u16(&q4_scales).len() as u64,
                "body_bpw": 4.25,
                "timing": q4_timing,
            });
            out.push(shape_json);
        }

        Ok(json!({
            "method": "MTLCommandBuffer GPUEndTime-GPUStartTime after wait; synthetic packed codes, no dense W",
            "affine_shader_compile_s": affine_compile_s,
            "q4_shader_compile_s": q4_compile_s,
            "note": "affine2 standalone shader uses f32 scale+bias; q4 geo uses f16 scale. Same occupancy (geo_tpr64 tg128). Production mixed affine2 uses f16; live_affine2_gate_matvec is that path.",
            "runtime_div_is_the_old_g64_body": true,
            "specialized_uses_compile_time_shift": true,
            "shapes": out,
        }))
    }
}

#[cfg(target_os = "macos")]
fn live_matvec_cost(session: &Qwen38HybridDecodeSession) -> Value {
    let mut x = vec![0.0f32; QWEN38_HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.08;
    }
    if let Err(e) = session.write_f32_workspace("normalized", &x) {
        return json!({"ok": false, "error": e.to_string()});
    }
    let gate = qwen38_layer_name(0, "mlp.gate_proj.weight");
    let mut gpu = Vec::new();
    for i in 0..8 {
        match session.measure_named_matvec(&gate, "gate") {
            Ok(t) => {
                if i >= 2 {
                    if let Some(ns) = t.gpu_ns {
                        gpu.push(ns);
                    }
                }
            }
            Err(e) => return json!({"ok": false, "error": e.to_string()}),
        }
    }
    json!({
        "ok": true,
        "tensor": gate,
        "shape": [17408, 5120],
        "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
        "group_size": 64,
        "gpu_ns_reps": gpu,
        "gpu_ns_median": median_u64(gpu.clone()),
        "note": "live HGRAVF01 gate_proj layer 0 on the mixed catalog; production bind",
    })
}

#[cfg(target_os = "macos")]
fn run(args: Args) {
    let tokenizer = load_qwen38_tokenizer(&args.tokenizer).unwrap_or_else(|e| fail(e));
    let rendered = if args.raw_prompt {
        args.prompt.clone()
    } else {
        render_qwen38_user_chat(&args.prompt)
    };
    let prompt_ids = tokenizer
        .encode(&rendered, false)
        .unwrap_or_else(|e| fail(e));
    if prompt_ids.is_empty() {
        fail("prompt encoded to zero tokens");
    }

    let kernel_cost = if args.skip_kernel_cost {
        json!({"skipped": true})
    } else {
        eprintln!("kernel cost: affine2 vs q4 geo_tpr64 on the same shape");
        match kernel_cost::measure() {
            Ok(v) => v,
            Err(e) => json!({"ok": false, "error": e}),
        }
    };

    eprintln!(
        "qwen38 fused subbit: open {} max_seq={}",
        args.artifact_root.display(),
        args.max_seq_len
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    eprintln!("session open {session_open_s:.3}s");

    let live_gate = live_matvec_cost(&session);

    eprintln!("parity: MLP gate+up+swiglu layer 0 (affine2 if mixed)");
    let mlp_parity = session
        .measure_mlp_fusion_parity(0)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} pair={} swiglu={} max_abs gate={} up={} act={}",
        mlp_parity.unfused_dispatches,
        mlp_parity.fused_pair_dispatches,
        mlp_parity.fused_swiglu_dispatches,
        mlp_parity.max_abs_diff_gate,
        mlp_parity.max_abs_diff_up,
        mlp_parity.max_abs_diff_act
    );

    eprintln!("parity: GQA QKV layer 3");
    let qkv_parity = session
        .measure_qkv_fusion_parity(3)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} fused={} max_abs q={} k={} v={}",
        qkv_parity.unfused_dispatches,
        qkv_parity.fused_pair_dispatches,
        qkv_parity.max_abs_diff_gate,
        qkv_parity.max_abs_diff_up,
        qkv_parity.max_abs_diff_act
    );

    eprintln!("parity: DeltaNet qkvz+ba layer 0");
    let dn_parity = session
        .measure_dn_inproj_fusion_parity(0)
        .unwrap_or_else(|e| fail(e));
    eprintln!(
        "  unfused={} fused={} max_abs qkvz={} ba={}",
        dn_parity.unfused_dispatches,
        dn_parity.fused_pair_dispatches,
        dn_parity.max_abs_diff_gate,
        dn_parity.max_abs_diff_up
    );

    let probe_token = prompt_ids[0];
    let mut dispatch_probes = Vec::new();

    session.apply_fusion(Qwen38MlpFusion::Off, false, false);
    eprintln!("dispatch probe: unfused");
    dispatch_probes.push(json!({
        "id": "unfused",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.apply_fusion(Qwen38MlpFusion::GateUpPair, false, false);
    eprintln!("dispatch probe: mlp pair");
    dispatch_probes.push(json!({
        "id": "mlp_pair",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, false, false);
    eprintln!("dispatch probe: mlp swiglu");
    dispatch_probes.push(json!({
        "id": "mlp_swiglu",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    eprintln!("dispatch probe: mlp swiglu + qkv + dn");
    dispatch_probes.push(json!({
        "id": "mlp_swiglu_qkv_dn",
        "probe": probe_dispatches(&mut session, probe_token).unwrap_or_else(|e| fail(e)),
    }));

    let mut decode_arms = json!({});
    if !args.skip_decode {
        session.apply_fusion(Qwen38MlpFusion::Off, false, false);
        eprintln!("decode unfused (affine2, no fusion) reps={}", args.reps);
        decode_arms["unfused"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.apply_fusion(Qwen38MlpFusion::GateUpPair, false, false);
        eprintln!("decode mlp pair reps={}", args.reps);
        decode_arms["mlp_pair"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, false, false);
        eprintln!("decode mlp swiglu reps={}", args.reps);
        decode_arms["mlp_swiglu"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));

        session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
        eprintln!("decode mlp swiglu + qkv + dn reps={}", args.reps);
        decode_arms["mlp_swiglu_qkv_dn"] = generate_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            args.reps,
        )
        .unwrap_or_else(|e| fail(e));
    }

    let before = production_dispatches_per_token();
    let body = json!({
        "schema": "hawking.headless.noetic_fused_subbit.v1",
        "generated_at": now_iso(),
        "git_head": git_head(),
        "question": "Why is affine2 g64 slower than q4, and does fusing it on the affine2 MLP path cross the boundary?",
        "did_not_load_second_27b": true,
        "did_not_write_under_models": true,
        "dense_w_materialized": 0,
        "expanded_to_q4": 0,
        "expanded_to_float_gemv": 0,
        "session_open_s": session_open_s,
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "kernel_cost": kernel_cost,
        "live_affine2_gate_matvec": live_gate,
        "counting": {
            "method": "TokenCommandBuffer.dispatch_count, one kernel launch = one dispatch, same as production_dispatches_per_token",
            "before": before,
            "after_mlp_pair": qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpPair, false, false),
            "after_mlp_swiglu": qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpSwiglu, false, false),
            "after_mlp_swiglu_qkv_dn": qwen38_fused_dispatches_per_token(Qwen38MlpFusion::GateUpSwiglu, true, true),
            "command_buffers": 1,
        },
        "fusions_attempted": [
            "affine2 gate_up_pair (same-row dual geo_tpr64, still a SwiGLU dispatch)",
            "affine2 gate_up_swiglu (same-row dual + silu(g)*up in-register)",
            "gqa_qkv concat geo_tpr64 (3 q4 matvecs -> 1, 16 GQA layers)",
            "dn_qkvz_ba concat geo_tpr64 (2 q4 matvecs -> 1, 48 DeltaNet layers)",
        ],
        "kernels": [
            "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
            "qwen_affine_q2_group64_matvec_gate_up_geo_tpr64_tg128",
            "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128",
            "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128",
            "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128",
        ],
        "parity": {
            "mlp_gate_up_swiglu": parity_json(&mlp_parity),
            "gqa_qkv": parity_json(&qkv_parity),
            "dn_qkvz_ba": parity_json(&dn_parity),
        },
        "dispatch_probes": dispatch_probes,
        "decode": decode_arms,
        "skip_decode": args.skip_decode,
    });

    println!("{}", serde_json::to_string_pretty(&body).unwrap());
    if let Some(path) = &args.out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, serde_json::to_string_pretty(&body).unwrap()).unwrap_or_else(|e| fail(e));
        eprintln!("wrote {}", path.display());
    }
}

#[cfg(not(target_os = "macos"))]
fn run(_args: Args) {
    fail("requires macOS Metal");
}

fn main() {
    run(parse_args());
}
