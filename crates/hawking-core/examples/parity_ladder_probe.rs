//! Parity-ladder probe: strongest surface first, not a complete-token A/B.
//!
//! fold_addqx_token_ab already timed the complete token and counted the
//! 22309-byte layer-0 gate mismatch. This example does not re-run that A/B.
//! It dumps the surfaces the ladder needs to *judge* that candidate:
//!   layer-0 named-matvec buffers with magnitude (not a byte count),
//!   last-layer MLP / hidden / logits on real prompts,
//!   KL and top-k (not argmax),
//!   a cheap greedy generate that would notice a behavioural change.
//!
//! qwen38 is dense: there are no route ids / selected experts to dump.
//!
//! ```text
//! cargo build --profile release-fast -p hawking-core \
//!   --example parity_ladder_probe
//! ./tools/gpu_lane_lock.sh g022parity \
//!   workspace/ops/build/rust/release-fast/examples/parity_ladder_probe \
//!   --artifact-root ~/noetic/NOETIC_PARENT_A \
//!   --tokenizer ~/noetic/NOETIC_PARENT_A/tokenizer.json \
//!   --out /tmp/parity_ladder_probe.json
//! ```

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_geometry::{
    qwen38_layer_name, QWEN38_HIDDEN, QWEN38_INTERMEDIATE, QWEN38_VOCAB,
};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen38_hybrid_decode::{
    load_qwen38_tokenizer, render_qwen38_user_chat, Affine2Geo, Qwen38DeltaNetStateKernel,
    Qwen38HybridDecodeSession, Qwen38MlpFusion, QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX,
    QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL, QWEN38_AFFINE_Q2_FOLD_ADDQX, QWEN38_AFFINE_Q2_GEO_TPR64,
};

fn usage() -> &'static str {
    "usage: parity_ladder_probe --artifact-root DIR --tokenizer PATH \
        [--max-new-tokens N] [--max-seq-len N] [--out FILE]"
}

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("parity_ladder_probe: {message}");
    process::exit(2);
}

struct Args {
    artifact_root: PathBuf,
    tokenizer: PathBuf,
    max_new_tokens: usize,
    max_seq_len: usize,
    out: Option<PathBuf>,
}

struct PromptSpec {
    id: &'static str,
    domain: &'static str,
    text: &'static str,
}

const PROMPTS: &[PromptSpec] = &[
    PromptSpec {
        id: "reasoning_add",
        domain: "reasoning",
        text: "2 + 2 =",
    },
    PromptSpec {
        id: "code_fib",
        domain: "code",
        text: "def fib(n):",
    },
    PromptSpec {
        id: "prose_capital",
        domain: "plain-prose",
        text: "The capital of France is",
    },
    PromptSpec {
        id: "instruction_json",
        domain: "tool-calling",
        text: "Reply with a JSON object with keys name and age for Ada Lovelace.",
    },
];

fn parse_args() -> Args {
    let mut artifact_root = None;
    let mut tokenizer = None;
    let mut max_new_tokens = 8usize;
    let mut max_seq_len = 128usize;
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
            "--out" => out = Some(PathBuf::from(args.next().unwrap_or_else(|| fail(usage())))),
            other => fail(format!("unknown {other}; {}", usage())),
        }
    }
    if max_new_tokens == 0 {
        fail("--max-new-tokens must be > 0");
    }
    Args {
        artifact_root: artifact_root.unwrap_or_else(|| fail(usage())),
        tokenizer: tokenizer.unwrap_or_else(|| fail(usage())),
        max_new_tokens,
        max_seq_len,
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
        "note": "parity probe, not a complete-token A/B; absolute times are under load",
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

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn ulp_u32(a: f32, b: f32) -> u32 {
    a.to_bits().abs_diff(b.to_bits())
}

fn characterize_f32(inc: &[f32], cand: &[f32], compared_against: &str) -> Value {
    let n = inc.len().min(cand.len());
    if n == 0 || inc.len() != cand.len() {
        return json!({
            "compared_against": compared_against,
            "n_floats_compared": n,
            "n_bytes_compared": n * 4,
            "error": "length mismatch or empty",
            "incumbent_len": inc.len(),
            "candidate_len": cand.len(),
        });
    }
    let ab = f32_le_bytes(inc);
    let bb = f32_le_bytes(cand);
    let mut n_mismatch_bytes = 0usize;
    let mut first_byte = None;
    for i in 0..ab.len() {
        if ab[i] != bb[i] {
            n_mismatch_bytes += 1;
            if first_byte.is_none() {
                first_byte = Some(i);
            }
        }
    }
    let mut n_float = 0usize;
    let mut first_float = None;
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_inc_sq = 0.0f64;
    let mut sum_cand_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_ulp = 0u32;
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    let mut n_sign = 0usize;
    let mut ulp_hist = [0usize; 9];
    let mut abs_hist = [0usize; 10];
    let mut median_ulps: Vec<u32> = Vec::new();
    let mut samples: Vec<Value> = Vec::new();
    for i in 0..n {
        let a = inc[i];
        let b = cand[i];
        if a.is_nan() || b.is_nan() {
            n_nan += 1;
        }
        if a.is_infinite() || b.is_infinite() {
            n_inf += 1;
        }
        let da = a as f64;
        let db = b as f64;
        let diff = da - db;
        let ad = diff.abs();
        sum_abs += ad;
        sum_sq += diff * diff;
        sum_inc_sq += da * da;
        sum_cand_sq += db * db;
        dot += da * db;
        if ad > max_abs {
            max_abs = ad;
        }
        let bits_differ = a.to_bits() != b.to_bits();
        if bits_differ {
            n_float += 1;
            if first_float.is_none() {
                first_float = Some(i);
            }
            let u = ulp_u32(a, b);
            if u > max_ulp {
                max_ulp = u;
            }
            median_ulps.push(u);
            if samples.len() < 12 {
                samples.push(json!({
                    "index": i,
                    "incumbent": a,
                    "candidate": b,
                    "abs": ad,
                    "ulp": u,
                }));
            }
            if a != 0.0 && b != 0.0 && a.signum() != b.signum() {
                n_sign += 1;
            }
            let bucket = if u == 0 {
                0
            } else if u == 1 {
                1
            } else if u <= 3 {
                2
            } else if u <= 7 {
                3
            } else if u <= 15 {
                4
            } else if u <= 63 {
                5
            } else if u <= 255 {
                6
            } else if u <= 1023 {
                7
            } else {
                8
            };
            ulp_hist[bucket] += 1;
        } else {
            ulp_hist[0] += 1;
        }
        let abucket = if ad == 0.0 {
            0
        } else if ad <= 1e-8 {
            1
        } else if ad <= 1e-7 {
            2
        } else if ad <= 1e-6 {
            3
        } else if ad <= 1e-5 {
            4
        } else if ad <= 1e-4 {
            5
        } else if ad <= 1e-3 {
            6
        } else if ad <= 1e-2 {
            7
        } else if ad <= 1e-1 {
            8
        } else {
            9
        };
        abs_hist[abucket] += 1;
    }
    let n_f = n as f64;
    let rel_l2 = if sum_inc_sq == 0.0 {
        if sum_sq == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        sum_sq.sqrt() / sum_inc_sq.sqrt()
    };
    let cosine = if sum_inc_sq == 0.0 || sum_cand_sq == 0.0 {
        f64::NAN
    } else {
        dot / (sum_inc_sq.sqrt() * sum_cand_sq.sqrt())
    };
    median_ulps.sort_unstable();
    let median_ulp = if median_ulps.is_empty() {
        0
    } else {
        median_ulps[median_ulps.len() / 2]
    };
    let rms_inc = (sum_inc_sq / n_f).sqrt();
    let rms_diff = (sum_sq / n_f).sqrt();
    let bit_identical = n_mismatch_bytes == 0 && n_float == 0;
    let cause = if bit_identical {
        "BIT_IDENTICAL"
    } else if n_nan > 0 || n_inf > 0 || rel_l2 > 0.03 || (!cosine.is_nan() && cosine < 0.999) {
        "DIFFERENT_COMPUTATION"
    } else {
        "SOURCE_ORDER_FMA_ASSOCIATION"
    };
    json!({
        "compared_against": compared_against,
        "n_bytes_compared": ab.len(),
        "n_mismatch_bytes": n_mismatch_bytes,
        "first_mismatch_byte": first_byte,
        "n_floats_compared": n,
        "n_float_mismatch": n_float,
        "float_mismatch_fraction": (n_float as f64) / n_f,
        "first_mismatch_float": first_float,
        "max_abs": max_abs,
        "mean_abs": sum_abs / n_f,
        "rms_diff": rms_diff,
        "rel_l2": rel_l2,
        "cosine": cosine,
        "max_ulp": max_ulp,
        "median_ulp_mismatch": median_ulp,
        "incumbent_rms": rms_inc,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_sign_flips": n_sign,
        "abs_histogram": {
            "eq0": abs_hist[0],
            "le_1e-8": abs_hist[1],
            "le_1e-7": abs_hist[2],
            "le_1e-6": abs_hist[3],
            "le_1e-5": abs_hist[4],
            "le_1e-4": abs_hist[5],
            "le_1e-3": abs_hist[6],
            "le_1e-2": abs_hist[7],
            "le_1e-1": abs_hist[8],
            "gt_1e-1": abs_hist[9],
        },
        "ulp_histogram": {
            "eq0": ulp_hist[0],
            "eq1": ulp_hist[1],
            "2_3": ulp_hist[2],
            "4_7": ulp_hist[3],
            "8_15": ulp_hist[4],
            "16_63": ulp_hist[5],
            "64_255": ulp_hist[6],
            "256_1023": ulp_hist[7],
            "ge_1024": ulp_hist[8],
        },
        "samples": samples,
        "bit_identical": bit_identical,
        "cause": cause,
        "incumbent_fnv1a64": format!("{:016x}", fnv1a64(&ab)),
        "candidate_fnv1a64": format!("{:016x}", fnv1a64(&bb)),
    })
}

fn log_sum_exp(logits: &[f32]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &v in logits {
        let d = v as f64;
        if d > max {
            max = d;
        }
    }
    let mut sum = 0.0;
    for &v in logits {
        sum += (v as f64 - max).exp();
    }
    max + sum.ln()
}

fn kl_from_logits(inc: &[f32], cand: &[f32]) -> f64 {
    let n = inc.len().min(cand.len());
    let lp_z = log_sum_exp(&inc[..n]);
    let lq_z = log_sum_exp(&cand[..n]);
    let mut kl = 0.0;
    for i in 0..n {
        let lp = inc[i] as f64 - lp_z;
        let lq = cand[i] as f64 - lq_z;
        let p = lp.exp();
        kl += p * (lp - lq);
    }
    kl
}

fn topk_ids(logits: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
    for (i, &v) in logits.iter().enumerate() {
        if best.len() < k {
            best.push((v, i));
            if best.len() == k {
                best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else if v > best[0].0 {
            best[0] = (v, i);
            best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    best.into_iter().map(|(_, i)| i).collect()
}

fn topk_agreement(inc: &[f32], cand: &[f32], k: usize) -> (f64, Vec<usize>, Vec<usize>) {
    let a = topk_ids(inc, k);
    let b = topk_ids(cand, k);
    let set_a: std::collections::HashSet<usize> = a.iter().copied().collect();
    let overlap = b.iter().filter(|i| set_a.contains(i)).count();
    let agree = if a.is_empty() {
        1.0
    } else {
        overlap as f64 / a.len() as f64
    };
    (agree, a, b)
}

fn logit_agreement(inc: &[f32], cand: &[f32], k: usize) -> Value {
    let kl = kl_from_logits(inc, cand);
    let (top, inc_top, cand_top) = topk_agreement(inc, cand, k);
    let inc_am = inc
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let cand_am = cand
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    json!({
        "kl_nats": kl,
        "top_k": k,
        "top_k_agreement": top,
        "argmax_agreement": if inc_am == cand_am { 1.0 } else { 0.0 },
        "argmax_is_not_parity": true,
        "incumbent_argmax": inc_am,
        "candidate_argmax": cand_am,
        "incumbent_top_k": inc_top,
        "candidate_top_k": cand_top,
        "n_rows": 1,
        "n_logits": inc.len().min(cand.len()),
        "parity_quantities": ["kl_nats", "top_k_agreement"],
    })
}

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
fn synthetic_x() -> Vec<f32> {
    // Same recipe as fold_addqx_token_ab.rs layer0_byte_compare. The 22309
    // byte count is on this x, not a new activation.
    let mut x = vec![0.0f32; QWEN38_HIDDEN];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.08;
    }
    x
}

#[cfg(target_os = "macos")]
fn layer0_named_matvec(session: &mut Qwen38HybridDecodeSession) -> Result<Value, String> {
    let x = synthetic_x();
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

    Ok(json!({
        "layer": 0,
        "x_recipe": "((i % 17) as f32) * 0.01 - 0.08, matching fold_addqx_token_ab.rs",
        "n_floats_gate_up": QWEN38_INTERMEDIATE,
        "n_floats_down": QWEN38_HIDDEN,
        "gate": characterize_f32(
            &gate_inc,
            &gate_f,
            "layer-0 mlp.gate_proj output after named matvec; same synthetic x as FOLD_ADDQX_AB",
        ),
        "up": characterize_f32(
            &up_inc,
            &up_f,
            "layer-0 mlp.up_proj output after named matvec; same synthetic x as FOLD_ADDQX_AB",
        ),
        "down": characterize_f32(
            &down_inc,
            &down_f,
            "layer-0 mlp.down_proj output after named matvec; same synthetic x as FOLD_ADDQX_AB",
        ),
        "compared_against":
            "production geo_tpr64 named-matvec output buffers on the live session, same x",
        "not_a_complete_token_ab": true,
    }))
}

#[cfg(target_os = "macos")]
struct ArmCapture {
    new_token_ids: Vec<u32>,
    text: String,
    fallbacks: u32,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    act: Vec<f32>,
    down: Vec<f32>,
    histogram: Vec<(String, u64)>,
    geo: String,
    launched: String,
}

#[cfg(target_os = "macos")]
fn run_arm(
    session: &mut Qwen38HybridDecodeSession,
    tokenizer: &hawking_core::tokenizer::Tokenizer,
    prompt_ids: &[u32],
    max_new: usize,
    fold: bool,
) -> Result<ArmCapture, String> {
    if fold {
        apply_fold_addqx(session);
    } else {
        apply_incumbent(session);
    }
    session.reset();
    let _ = session.drain_dispatched_kernel_names();
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }
    let mut next = 0u32;
    for &token in prompt_ids {
        let (sampled, _) = session.step(token).map_err(|e| e.to_string())?;
        next = sampled;
    }
    let hidden = session
        .read_f32_workspace("hidden", QWEN38_HIDDEN)
        .map_err(|e| e.to_string())?;
    let logits = session
        .read_f32_workspace("logits", QWEN38_VOCAB)
        .map_err(|e| e.to_string())?;
    let gate = session
        .read_f32_workspace("gate", QWEN38_INTERMEDIATE)
        .map_err(|e| e.to_string())?;
    let up = session
        .read_f32_workspace("up", QWEN38_INTERMEDIATE)
        .map_err(|e| e.to_string())?;
    let act = session
        .read_f32_workspace("act", QWEN38_INTERMEDIATE)
        .map_err(|e| e.to_string())?;
    let down = session
        .read_f32_workspace("down", QWEN38_HIDDEN)
        .map_err(|e| e.to_string())?;

    let mut new_ids = vec![next];
    while new_ids.len() < max_new {
        let (sampled, _) = session.step(next).map_err(|e| e.to_string())?;
        new_ids.push(sampled);
        next = sampled;
    }
    let histogram = session.dispatched_kernel_histogram();
    let text = tokenizer.decode(&new_ids, true).map_err(|e| e.to_string())?;
    Ok(ArmCapture {
        new_token_ids: new_ids,
        text,
        fallbacks: session.fallbacks,
        hidden,
        logits,
        gate,
        up,
        act,
        down,
        histogram,
        geo: session.affine2_geo.as_str().to_string(),
        launched: session.launched_gated_delta_kernel().to_string(),
    })
}

#[cfg(target_os = "macos")]
fn histogram_json(hist: &[(String, u64)]) -> Value {
    json!(hist
        .iter()
        .map(|(k, n)| json!({"kernel": k, "count": n}))
        .collect::<Vec<_>>())
}

#[cfg(target_os = "macos")]
fn kernel_count(hist: &[(String, u64)], name: &str) -> u64 {
    hist.iter()
        .filter(|(k, _)| k == name)
        .map(|(_, n)| *n)
        .sum()
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
    let load_start = concurrent_load();
    eprintln!(
        "parity_ladder_probe: open {} max_seq={} max_new={} (NOT a complete-token A/B)",
        args.artifact_root.display(),
        args.max_seq_len,
        args.max_new_tokens
    );
    let open_started = Instant::now();
    let mut session = Qwen38HybridDecodeSession::open(&args.artifact_root, args.max_seq_len)
        .unwrap_or_else(|e| fail(e));
    let session_open_s = open_started.elapsed().as_secs_f64();
    apply_incumbent(&mut session);
    eprintln!(
        "session open {session_open_s:.3}s dense_w={} theoretical={} launched={} geo={}",
        session.dense_w_materialized,
        session.theoretical_dispatches(),
        session.launched_gated_delta_kernel(),
        session.affine2_geo.as_str()
    );

    let layer0 = layer0_named_matvec(&mut session).unwrap_or_else(|e| fail(e));
    eprintln!(
        "layer0 gate mismatch_bytes={} rel_l2={} cause={}",
        layer0["gate"]["n_mismatch_bytes"],
        layer0["gate"]["rel_l2"],
        layer0["gate"]["cause"]
    );

    let mut live_prompts: Vec<Value> = Vec::new();
    for spec in PROMPTS {
        let rendered = render_qwen38_user_chat(spec.text);
        let prompt_ids = tokenizer
            .encode(&rendered, false)
            .unwrap_or_else(|e| fail(e));
        if prompt_ids.is_empty() {
            fail(format!("{} encoded to zero tokens", spec.id));
        }
        eprintln!(
            "prompt {} domain={} n_ids={} max_new={}",
            spec.id,
            spec.domain,
            prompt_ids.len(),
            args.max_new_tokens
        );
        let inc = run_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            false,
        )
        .unwrap_or_else(|e| fail(e));
        let fold = run_arm(
            &mut session,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            true,
        )
        .unwrap_or_else(|e| fail(e));
        let fold_swiglu = kernel_count(&fold.histogram, QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX);
        let inc_swiglu = kernel_count(&inc.histogram, QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL);
        if fold_swiglu == 0 {
            fail(format!(
                "{} fold_addqx arm did not dispatch {QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX}",
                spec.id
            ));
        }
        if kernel_count(&fold.histogram, QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL) > 0 {
            fail(format!(
                "{} fold_addqx arm still dispatched production swiglu",
                spec.id
            ));
        }
        let logits = logit_agreement(&inc.logits, &fold.logits, 5);
        let hidden = characterize_f32(
            &inc.hidden,
            &fold.hidden,
            "hidden residual after last prompt token, same prefix",
        );
        let last_layer = json!({
            "gate": characterize_f32(&inc.gate, &fold.gate, "workspace gate after last prompt token (fused path may not refresh this)"),
            "up": characterize_f32(&inc.up, &fold.up, "workspace up after last prompt token"),
            "act": characterize_f32(&inc.act, &fold.act, "workspace act (silu(gate)*up) after last prompt token"),
            "down": characterize_f32(&inc.down, &fold.down, "workspace down after last prompt token"),
        });
        let token_ids_identical = inc.new_token_ids == fold.new_token_ids;
        eprintln!(
            "  {} tokens_identical={} kl={} top5={} fold_swiglu={} inc_swiglu={}",
            spec.id,
            token_ids_identical,
            logits["kl_nats"],
            logits["top_k_agreement"],
            fold_swiglu,
            inc_swiglu
        );
        live_prompts.push(json!({
            "id": spec.id,
            "domain": spec.domain,
            "prompt": spec.text,
            "rendered_prompt": rendered,
            "prompt_ids": prompt_ids,
            "incumbent_token_ids": inc.new_token_ids,
            "candidate_token_ids": fold.new_token_ids,
            "incumbent_text": inc.text,
            "candidate_text": fold.text,
            "token_ids_identical": token_ids_identical,
            "fallbacks": inc.fallbacks + fold.fallbacks,
            "logits": logits,
            "hidden": hidden,
            "last_layer_mlp": last_layer,
            "incumbent": {
                "new_token_ids": inc.new_token_ids,
                "generated_text": inc.text,
                "fallbacks": inc.fallbacks,
                "affine2_geo": inc.geo,
                "launched_gated_delta_kernel": inc.launched,
                "kernel_histogram": histogram_json(&inc.histogram),
                "swiglu_count": inc_swiglu,
            },
            "fold_addqx": {
                "new_token_ids": fold.new_token_ids,
                "generated_text": fold.text,
                "fallbacks": fold.fallbacks,
                "affine2_geo": fold.geo,
                "launched_gated_delta_kernel": fold.launched,
                "kernel_histogram": histogram_json(&fold.histogram),
                "swiglu_count": fold_swiglu,
                "down_kernel": QWEN38_AFFINE_Q2_FOLD_ADDQX,
            },
            "agreement": logits,
        }));
    }

    let load_end = concurrent_load();
    let body = json!({
        "schema": "hawking.future.parity_ladder.probe.v1",
        "git_head": git_head(),
        "artifact_root": args.artifact_root,
        "tokenizer": args.tokenizer,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "session_open_s": session_open_s,
        "dense_w_materialized": session.dense_w_materialized,
        "not_a_complete_token_ab": true,
        "does_not_rerun_fold_addqx_ab": true,
        "organ_has_routes": false,
        "organ_has_selected_experts": false,
        "dense_reason": "qwen38 is dense Qwen3.5-27B hybrid; geometry refuses MoE keys",
        "expected_kernels": {
            "incumbent_gate_up_swiglu": QWEN38_AFFINE_GATE_UP_SWIGLU_KERNEL,
            "incumbent_down": QWEN38_AFFINE_Q2_GEO_TPR64,
            "fold_addqx_gate_up_swiglu": QWEN38_AFFINE_GATE_UP_SWIGLU_FOLD_ADDQX,
            "fold_addqx_down": QWEN38_AFFINE_Q2_FOLD_ADDQX,
        },
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
        "concurrent_load_start": load_start,
        "concurrent_load": load_end,
        "layer0_named_matvec": layer0,
        "live_prompts": live_prompts,
        "capability": live_prompts,
        "x_recipe": "((i % 17) as f32) * 0.01 - 0.08",
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
