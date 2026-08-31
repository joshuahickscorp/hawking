//! Complete-token A/B: post-widen_f4 incumbent vs fold_addqx on the real
//! decode path.
//!
//! fold_addqx is the bit-identical MLP decode cheapen (MLP_DECODE_CHEAPEN
//! 370.9 GB/s, 1.127x on one layer). That number sat at DIRTY_DIAGNOSTIC
//! until this A/B ran it on encode_mlp / encode_fused_gate_up and reported
//! the complete token. Incumbent is the post-widen_f4 580-graph, which is
//! also the new baseline measurement — PATH_TO_71's 28.722 ms is stale.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example fold_addqx_token_ab
//! ./tools/gpu_lane_lock.sh g1address \
//!   workspace/ops/build/rust/release-fast/examples/fold_addqx_token_ab \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/noetic/NOETIC_PARENT_A/tokenizer.json \
//!   --reps 7 --max-new-tokens 32 \
//!   --out receipts/future/_FOLD_ADDQX_AB_raw.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_geometry::{
    qwen38_layer_name, QWEN38_HIDDEN, QWEN38_INTERMEDIATE,
};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    generate_greedy, load_qwen38_tokenizer, render_qwen38_user_chat, Affine2Geo,
    Qwen38DeltaNetStateKernel, Qwen38GenerateResult, Qwen38HybridDecodeSession, Qwen38MlpFusion,
    QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX, QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL,
    QWEN38_AFFINE_Q2_FOLD_ADDQX, QWEN38_AFFINE_Q2_GEO_TPR64, QWEN38_DN_STATE_F4_KERNEL,
};

fn usage() -> &'static str {
    "usage: fold_addqx_token_ab --artifact-root DIR --tokenizer PATH \
        [--prompt TEXT] [--raw-prompt] [--max-new-tokens N] [--max-seq-len N] \
        [--reps N] [--warmup N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("fold_addqx_token_ab: {message}");
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
    warmup: usize,
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
    let mut max_new_tokens = 32usize;
    let mut max_seq_len = 128usize;
    let mut reps = 7usize;
    let mut warmup = 1usize;
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
            "--warmup" => {
                warmup = args
                    .next()
                    .unwrap_or_else(|| fail(usage()))
                    .parse()
                    .unwrap_or_else(|_| fail("--warmup"));
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if reps < 7 {
        fail("--reps must be >= 7 (S020 §37)");
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        prompt,
        raw_prompt,
        max_new_tokens,
        max_seq_len,
        reps,
        warmup,
        out,
    }
}

fn git_head() -> String {
    std::process::Command::new("git")
        .args(["--no-optional-locks", "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn cmd_stdout(args: &[&str]) -> String {
    std::process::Command::new(args[0])
        .args(&args[1..])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
}

fn concurrent_load() -> Value {
    let loadavg = cmd_stdout(&["sysctl", "-n", "vm.loadavg"])
        .trim()
        .to_string();
    let uptime = cmd_stdout(&["uptime"]).trim().to_string();
    json!({
        "loadavg": loadavg,
        "uptime": uptime,
        "note": "absolute ms are measured-under-load; the A/B ratio is back-to-back in this process",
    })
}

fn median_u64(mut v: Vec<u64>) -> Option<u64> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    Some(v[v.len() / 2])
}

fn timing_json(name: &str, gpu: &[u64], wait: &[u64], dispatches: u64, reps: usize) -> Value {
    json!({
        "name": name,
        "gpu_ns_reps": gpu,
        "gpu_ns_min": gpu.iter().copied().min(),
        "gpu_ns_median": median_u64(gpu.to_vec()),
        "gpu_ns_max": gpu.iter().copied().max(),
        "wait_ns_median": median_u64(wait.to_vec()),
        "dispatches": dispatches,
        "n_reps": reps,
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
        "dense_w_materialized": 0,
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn u32_le_bytes(ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn byte_compare_bytes(a: &[u8], b: &[u8], compared_against: &str) -> Value {
    let n = a.len().min(b.len());
    let mut n_mismatch = 0usize;
    let mut first = None;
    for i in 0..n {
        if a[i] != b[i] {
            n_mismatch += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    if a.len() != b.len() && first.is_none() {
        first = Some(n);
        n_mismatch += a.len().abs_diff(b.len());
    }
    json!({
        "compared_against": compared_against,
        "n_bytes_compared": n,
        "incumbent_len": a.len(),
        "candidate_len": b.len(),
        "n_mismatch_bytes": n_mismatch,
        "first_mismatch_index": first,
        "bit_identical": n_mismatch == 0 && a.len() == b.len(),
        "incumbent_fnv1a64": format!("{:016x}", fnv1a64(a)),
        "candidate_fnv1a64": format!("{:016x}", fnv1a64(b)),
    })
}

#[cfg(target_os = "macos")]
fn reps_cb(
    label: &str,
    reps: usize,
    mut once: impl FnMut() -> Result<hawking_core::metal::CommandBufferTiming, String>,
) -> Result<Value, String> {
    let mut gpu = Vec::new();
    let mut wait = Vec::new();
    let mut disp = 0u64;
    eprintln!("  {label} reps={reps}");
    for i in 0..reps {
        let t = once()?;
        let g = t
            .gpu_ns
            .ok_or_else(|| format!("{label}: driver did not expose GPUEndTime-GPUStartTime"))?;
        eprintln!("    rep{i} gpu={g} wait={} disp={}", t.wait_ns, t.dispatches);
        gpu.push(g);
        wait.push(t.wait_ns);
        disp = t.dispatches;
    }
    Ok(timing_json(label, &gpu, &wait, disp, reps))
}

#[cfg(target_os = "macos")]
fn complete_token_gpu_ns(result: &Qwen38GenerateResult) -> Vec<u64> {
    let start = result.prompt_len.saturating_sub(1);
    result
        .gpu_ns
        .get(start.min(result.gpu_ns.len())..)
        .unwrap_or(&[])
        .iter()
        .copied()
        .flatten()
        .collect()
}

#[cfg(target_os = "macos")]
fn complete_token_dispatches(result: &Qwen38GenerateResult) -> Vec<u64> {
    let start = result.prompt_len.saturating_sub(1);
    result
        .dispatches
        .get(start.min(result.dispatches.len())..)
        .unwrap_or(&[])
        .to_vec()
}

/// Post-widen_f4 sealed 580-graph. This IS the new incumbent baseline.
#[cfg(target_os = "macos")]
fn apply_incumbent(session: &mut Qwen38HybridDecodeSession) {
    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(false, false);
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::WidenF4);
    session.apply_affine2_geo(Affine2Geo::Tpr64);
}

#[cfg(target_os = "macos")]
fn apply_fold_addqx(session: &mut Qwen38HybridDecodeSession) {
    session.apply_fusion(Qwen38MlpFusion::GateUpSwiglu, true, true);
    session.set_fuse_add_rmsnorm(true, false);
    session.set_fuse_ba_delta(false, false);
    session.set_dn_state_kernel(Qwen38DeltaNetStateKernel::WidenF4);
    session.apply_affine2_geo(Affine2Geo::FoldAddqx);
}

#[cfg(target_os = "macos")]
fn generate_once(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
) -> Result<Value, String> {
    session.reset();
    let _ = session.drain_dispatched_kernel_names();
    let result = generate_greedy(session, prompt_ids, max_new).map_err(|e| e.to_string())?;
    let histogram = session.dispatched_kernel_histogram();
    let names = session.drain_dispatched_kernel_names();
    let text = result.decode_new(tokenizer).map_err(|e| e.to_string())?;
    let new_ids = result.new_tokens().to_vec();
    let complete = complete_token_gpu_ns(&result);
    let disp = complete_token_dispatches(&result);
    let launched = session.launched_gated_delta_kernel();
    Ok(json!({
        "generated_text": text,
        "new_token_ids": new_ids,
        "new_token_ids_le_bytes_fnv1a64": format!("{:016x}", fnv1a64(&u32_le_bytes(&new_ids))),
        "prompt_len": result.prompt_len,
        "decode_steps": result.decode_steps,
        "fallbacks": result.fallbacks,
        "dense_w_materialized": result.dense_w_materialized,
        "decode_wall_ns": result.decode_wall_ns,
        "prefill_wall_ns": result.prefill_wall_ns,
        "complete_token_gpu_ns": complete,
        "complete_token_gpu_ns_median": median_u64(complete.clone()),
        "complete_token_dispatches": disp,
        "complete_token_dispatches_last": disp.last().copied(),
        "theoretical_dispatches": session.theoretical_dispatches(),
        "launched_gated_delta_kernel": launched,
        "affine2_geo": session.affine2_geo.as_str(),
        "dn_state_kernel": session.dn_state_kernel.as_str(),
        "kernel_histogram": histogram.iter().map(|(k, n)| json!({"kernel": k, "count": n})).collect::<Vec<_>>(),
        "kernel_names": names,
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
    }))
}

#[cfg(target_os = "macos")]
fn layer0_byte_compare(session: &mut Qwen38HybridDecodeSession) -> Result<Value, String> {
    let mut x = vec![0.0f32; QWEN38_HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.08;
    }
    let gate_name = qwen38_layer_name(0, "mlp.gate_proj.weight");
    let up_name = qwen38_layer_name(0, "mlp.up_proj.weight");
    let down_name = qwen38_layer_name(0, "mlp.down_proj.weight");

    apply_incumbent(session);
    session
        .write_f32_workspace("normalized", &x)
        .map_err(|e| e.to_string())?;
    session
        .measure_named_matvec(&gate_name, "gate")
        .map_err(|e| e.to_string())?;
    let gate_inc = session
        .read_f32_workspace("gate", QWEN38_INTERMEDIATE)
        .map_err(|e| e.to_string())?;
    session
        .write_f32_workspace("normalized", &x)
        .map_err(|e| e.to_string())?;
    session
        .measure_named_matvec(&up_name, "up")
        .map_err(|e| e.to_string())?;
    let up_inc = session
        .read_f32_workspace("up", QWEN38_INTERMEDIATE)
        .map_err(|e| e.to_string())?;
    session
        .write_f32_workspace("act", &gate_inc)
        .map_err(|e| e.to_string())?;
    session
        .measure_named_matvec(&down_name, "down")
        .map_err(|e| e.to_string())?;
    let down_inc = session
        .read_f32_workspace("down", QWEN38_HIDDEN)
        .map_err(|e| e.to_string())?;

    apply_fold_addqx(session);
    session
        .write_f32_workspace("normalized", &x)
        .map_err(|e| e.to_string())?;
    session
        .measure_named_matvec(&gate_name, "gate")
        .map_err(|e| e.to_string())?;
    let gate_f = session
        .read_f32_workspace("gate", QWEN38_INTERMEDIATE)
        .map_err(|e| e.to_string())?;
    session
        .write_f32_workspace("normalized", &x)
        .map_err(|e| e.to_string())?;
    session
        .measure_named_matvec(&up_name, "up")
        .map_err(|e| e.to_string())?;
    let up_f = session
        .read_f32_workspace("up", QWEN38_INTERMEDIATE)
        .map_err(|e| e.to_string())?;
    session
        .write_f32_workspace("act", &gate_inc)
        .map_err(|e| e.to_string())?;
    session
        .measure_named_matvec(&down_name, "down")
        .map_err(|e| e.to_string())?;
    let down_f = session
        .read_f32_workspace("down", QWEN38_HIDDEN)
        .map_err(|e| e.to_string())?;

    let gate = byte_compare_bytes(
        &f32_le_bytes(&gate_inc),
        &f32_le_bytes(&gate_f),
        "layer-0 mlp.gate_proj output buffer after named matvec; not the probe",
    );
    let up = byte_compare_bytes(
        &f32_le_bytes(&up_inc),
        &f32_le_bytes(&up_f),
        "layer-0 mlp.up_proj output buffer after named matvec; not the probe",
    );
    let down = byte_compare_bytes(
        &f32_le_bytes(&down_inc),
        &f32_le_bytes(&down_f),
        "layer-0 mlp.down_proj output buffer after named matvec; not the probe",
    );
    let bit_identical = gate["bit_identical"].as_bool() == Some(true)
        && up["bit_identical"].as_bool() == Some(true)
        && down["bit_identical"].as_bool() == Some(true);
    Ok(json!({
        "layer": 0,
        "n_floats_gate_up": QWEN38_INTERMEDIATE,
        "n_floats_down": QWEN38_HIDDEN,
        "gate": gate,
        "up": up,
        "down": down,
        "bit_identical": bit_identical,
        "compared_against":
            "production geo_tpr64 named-matvec output buffers on the live session, same x",
        "note":
            "Token-level arithmetic identity is this byte comparison plus token-id bytes. Citing MLP_DECODE_CHEAPEN is refused as the identity proof.",
    }))
}

#[cfg(target_os = "macos")]
fn run(args: Args) {
    std::env::set_var("HAWKING_TRACE_DISPATCH", "1");
    std::env::set_var("HAWKING_QWEN_RESIDENCY", "1");
    std::env::set_var("HAWKING_QWEN38_IGNORE_EOS", "1");
    std::env::remove_var("HAWKING_QWEN38_FUSE_MLP");
    std::env::remove_var("HAWKING_QWEN38_FUSE_GQA_QKV");
    std::env::remove_var("HAWKING_QWEN38_FUSE_DN_INPROJ");
    std::env::remove_var("HAWKING_QWEN38_FUSE_ADD_RMSNORM");
    std::env::remove_var("HAWKING_QWEN38_FUSE_BA_DELTA");
    std::env::remove_var("HAWKING_QWEN38_DN_STATE");
    std::env::remove_var("HAWKING_AFFINE2_GEO");
    std::env::remove_var("HAWKING_QWEN38_FAST");
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

    let load_start = concurrent_load();
    eprintln!(
        "qwen38 fold_addqx token_ab: open {} max_seq={} max_new={} reps={} warmup={}",
        args.artifact_root.display(),
        args.max_seq_len,
        args.max_new_tokens,
        args.reps,
        args.warmup
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    apply_incumbent(&mut session);
    eprintln!(
        "session open {session_open_s:.3}s dense_w={} theoretical_580={} launched={} geo={}",
        session.dense_w_materialized,
        session.theoretical_dispatches(),
        session.launched_gated_delta_kernel(),
        session.affine2_geo.as_str()
    );

    let layer0 = layer0_byte_compare(&mut session).unwrap_or_else(|e| fail(e));
    eprintln!(
        "layer0 byte-compare bit_identical={}",
        layer0
            .get("bit_identical")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    );

    apply_incumbent(&mut session);
    let iso_inc = reps_cb("mlp_full_incumbent", args.reps, || {
        session
            .measure_isolated_mlp_full()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    apply_fold_addqx(&mut session);
    let iso_fold = reps_cb("mlp_full_fold_addqx", args.reps, || {
        session
            .measure_isolated_mlp_full()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    apply_incumbent(&mut session);
    let matvec_inc = reps_cb("mlp_matvecs_incumbent", args.reps, || {
        session
            .measure_isolated_mlp_matvecs()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));
    apply_fold_addqx(&mut session);
    let matvec_fold = reps_cb("mlp_matvecs_fold_addqx", args.reps, || {
        session
            .measure_isolated_mlp_matvecs()
            .map_err(|e| e.to_string())
    })
    .unwrap_or_else(|e| fail(e));

    let load_after_iso = concurrent_load();

    let mut incumbent_runs: Vec<Value> = Vec::new();
    let mut fold_runs: Vec<Value> = Vec::new();
    let total = args.warmup + args.reps;
    for i in 0..total {
        let warm = i < args.warmup;
        apply_incumbent(&mut session);
        eprintln!(
            "decode incumbent post-widen_f4 {} {i}/{total} geo={} launched={}",
            if warm { "warmup" } else { "rep" },
            session.affine2_geo.as_str(),
            session.launched_gated_delta_kernel()
        );
        let inc = generate_once(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
        )
        .unwrap_or_else(|e| fail(e));
        apply_fold_addqx(&mut session);
        eprintln!(
            "decode fold_addqx {} {i}/{total} geo={} launched={}",
            if warm { "warmup" } else { "rep" },
            session.affine2_geo.as_str(),
            session.launched_gated_delta_kernel()
        );
        let fold = generate_once(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
        )
        .unwrap_or_else(|e| fail(e));
        if !warm {
            incumbent_runs.push(inc);
            fold_runs.push(fold);
        }
    }

    let load_end = concurrent_load();
    let inc_medians: Vec<u64> = incumbent_runs
        .iter()
        .filter_map(|r| r.get("complete_token_gpu_ns_median").and_then(Value::as_u64))
        .collect();
    let fold_medians: Vec<u64> = fold_runs
        .iter()
        .filter_map(|r| r.get("complete_token_gpu_ns_median").and_then(Value::as_u64))
        .collect();

    let token_byte_compares: Vec<Value> = incumbent_runs
        .iter()
        .zip(fold_runs.iter())
        .map(|(inc, fold)| {
            let a: Vec<u32> = inc
                .get("new_token_ids")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_u64)
                        .map(|v| v as u32)
                        .collect()
                })
                .unwrap_or_default();
            let b: Vec<u32> = fold
                .get("new_token_ids")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_u64)
                        .map(|v| v as u32)
                        .collect()
                })
                .unwrap_or_default();
            byte_compare_bytes(
                &u32_le_bytes(&a),
                &u32_le_bytes(&b),
                "complete-token new_token_ids as little-endian u32 bytes; not the probe",
            )
        })
        .collect();

    let body = json!({
        "schema": "hawking.future.fold_addqx_ab.raw.v1",
        "git_head": git_head(),
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "rendered_prompt": rendered,
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "reps": args.reps,
        "warmup": args.warmup,
        "session_open_s": session_open_s,
        "dense_w_materialized": session.dense_w_materialized,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_ms_are_measured_under_load": true,
        "incumbent_is_post_widen_f4_baseline": true,
        "production_fusions": {
            "mlp": "GateUpSwiglu",
            "fuse_gqa_qkv": true,
            "fuse_dn_inproj": true,
            "fuse_add_rmsnorm": true,
            "fuse_ba_delta": false,
            "dn_state_kernel": "widen_f4",
            "affine2_geo_incumbent": "tpr64",
            "affine2_geo_candidate": "fold_addqx",
        },
        "expected_kernels": {
            "incumbent_gate_up_swiglu": QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL,
            "incumbent_down": QWEN38_AFFINE_Q2_GEO_TPR64,
            "fold_addqx_gate_up_swiglu": QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX,
            "fold_addqx_down": QWEN38_AFFINE_Q2_FOLD_ADDQX,
            "dn_state": QWEN38_DN_STATE_F4_KERNEL,
        },
        "concurrent_load_start": load_start,
        "concurrent_load_after_isolated": load_after_iso,
        "concurrent_load": load_end,
        "layer0_byte_compare": layer0,
        "isolated_mlp_full": {
            "incumbent": iso_inc,
            "fold_addqx": iso_fold,
        },
        "isolated_mlp_matvecs": {
            "incumbent": matvec_inc,
            "fold_addqx": matvec_fold,
        },
        "decode": {
            "interleaved": true,
            "incumbent": incumbent_runs,
            "fold_addqx": fold_runs,
            "incumbent_complete_token_gpu_ns_median_reps": inc_medians,
            "fold_addqx_complete_token_gpu_ns_median_reps": fold_medians,
            "incumbent_complete_token_gpu_ns_median": median_u64(inc_medians.clone()),
            "fold_addqx_complete_token_gpu_ns_median": median_u64(fold_medians.clone()),
            "token_id_byte_compare": token_byte_compares,
        },
        "gpu_timestamp_authority":
            "completed MTLCommandBuffer GPUStartTime/GPUEndTime after wait; never a CPU-wait proxy",
    });
    let text = serde_json::to_string_pretty(&body).unwrap_or_else(|e| fail(e));
    println!("{text}");
    if let Some(path) = &args.out {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(path, format!("{text}\n")).unwrap_or_else(|e| fail(e));
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
