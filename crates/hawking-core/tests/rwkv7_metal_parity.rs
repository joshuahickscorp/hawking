//! RWKV-7 GPU-decode parity gate: the Metal WKV-7 path must match the
//! PARITY-VALIDATED CPU reference `forward_token` bit-for-bit within f32
//! tolerance.
//!
//! The CPU `rwkv7.rs::forward_token` is the correctness oracle (it is itself
//! validated F32 48/48 / Q4_K 40/40 vs llama.cpp). The GPU path uploads the
//! SAME f32-dequantized weights and runs the SAME op order, so the only residual
//! is f32 reduction-order rounding — the two must agree to a tight tolerance and
//! produce identical argmax tokens.
//!
//! macOS/Metal-gated: skips cleanly (passes) when no Metal GPU is present or the
//! model weights are absent, so CI on non-Metal hosts is green.
//!
//! Two checks:
//!   1. `rwkv7_gpu_matches_cpu_logits` — feed an identical real token trajectory
//!      (the committed prompt + its CPU greedy continuation) through a fresh CPU
//!      state and a fresh GPU state in lockstep; assert per-step argmax match and
//!      max-abs logit diff under tolerance for >=32 steps.
//!   2. `rwkv7_gpu_greedy_trajectory_matches_cpu` — let each path drive its OWN
//!      greedy argmax for 32 steps from the same prompt; the produced id
//!      sequences must be identical (a self-consistent GPU decode == CPU decode).

#![cfg(target_os = "macos")]

use hawking_core::model::rwkv7::RwkvSeven;
use hawking_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};

/// Tolerance on the per-step max-abs logit difference (GPU vs CPU). The two
/// share f32 weights; only reduction order differs, so the gap is small. The
/// LM-head logits reach magnitudes ~O(10–30), so a few 1e-2 of slack is ample
/// headroom while still catching any real kernel bug (which diverges by O(1+)).
const LOGIT_TOL: f32 = 0.05;

fn read_ids(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read fixture {path:?}: {e}"))
        .split_whitespace()
        .map(|t| t.parse::<u32>().expect("fixture id parse"))
        .collect()
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/rwkv7")
        .join(name)
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i as u32;
        }
    }
    bi
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Locate the shipped Q4_K rwkv7-0.4B GGUF. Tries (in order): an explicit
/// `HAWKING_RWKV7_GGUF` override, the in-tree `../../models` path (normal
/// checkout / CI), then walks up from the manifest dir looking for a `models/`
/// dir (covers git-worktree layouts where `models/` lives in the main checkout).
fn locate_q4k() -> Option<PathBuf> {
    const REL: &str = "models/rwkv7-04/rwkv7-0.4B-world.Q4_K_M.gguf";
    if let Ok(p) = std::env::var("HAWKING_RWKV7_GGUF") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let direct = PathBuf::from("../..").join(REL);
    if direct.exists() {
        return Some(direct);
    }
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let cand = dir.join(REL);
        if cand.exists() {
            return Some(cand);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn load_model() -> Option<RwkvSeven> {
    // Prefer an F32 GGUF when available (tightest parity), else the shipped Q4_K.
    let f32_path = std::env::var("HAWKING_RWKV7_F32_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/rwkv_ref/rwkv7-04-f32.gguf"));
    let weights = if f32_path.exists() {
        f32_path
    } else if let Some(q4k) = locate_q4k() {
        q4k
    } else {
        eprintln!("skipping rwkv7_metal_parity: no rwkv7 weights (F32 or Q4_K) found");
        return None;
    };
    let engine = RwkvSeven::load(&weights, EngineConfig::default()).expect("load rwkv7");
    if !engine.has_gpu() {
        eprintln!("skipping rwkv7_metal_parity: Metal GPU not available");
        return None;
    }
    Some(engine)
}

/// Lockstep: identical input trajectory through fresh CPU and GPU states.
#[test]
fn rwkv7_gpu_matches_cpu_logits() {
    let Some(mut engine) = load_model() else {
        return;
    };

    // Real trajectory: prompt + its CPU greedy continuation (>= 45 ids).
    let mut input = read_ids(&fixture("capital_france_q4k.prompt_ids"));
    input.extend(read_ids(&fixture("capital_france_q4k.gen_ids")));
    assert!(input.len() >= 32, "need >=32 steps, got {}", input.len());

    // CPU pass (oracle): fresh state, collect logits per step.
    engine.reset_kv_for_test();
    let mut cpu_logits = Vec::with_capacity(input.len());
    for &t in &input {
        cpu_logits.push(engine.forward_token(t).expect("cpu forward"));
    }

    // GPU pass: fresh state, same inputs.
    engine.reset_kv_for_test();
    let mut worst = 0.0f32;
    let mut worst_step = 0usize;
    let mut argmax_mismatches = 0usize;
    for (step, &t) in input.iter().enumerate() {
        let gl = engine.forward_token_gpu(t).expect("gpu forward");
        let cl = &cpu_logits[step];
        assert_eq!(gl.len(), cl.len(), "logit width mismatch at step {step}");
        let d = max_abs_diff(&gl, cl);
        if d > worst {
            worst = d;
            worst_step = step;
        }
        let (ag, ac) = (argmax(&gl), argmax(cl));
        if ag != ac {
            argmax_mismatches += 1;
            eprintln!("step {step}: argmax GPU={ag} CPU={ac} (max|Δ|={d:.4})");
        }
    }
    eprintln!(
        "rwkv7 GPU↔CPU parity: {} steps, worst max|Δlogit|={:.5} @step {}, argmax mismatches={}",
        input.len(),
        worst,
        worst_step,
        argmax_mismatches
    );
    assert_eq!(
        argmax_mismatches, 0,
        "GPU decode argmax must match CPU oracle every step ({} mismatches)",
        argmax_mismatches
    );
    assert!(
        worst < LOGIT_TOL,
        "GPU↔CPU max-abs logit diff {worst:.5} exceeds tol {LOGIT_TOL} (worst @step {worst_step})"
    );
}

/// Each path drives its own greedy argmax; trajectories must be identical.
#[test]
fn rwkv7_gpu_greedy_trajectory_matches_cpu() {
    let Some(mut engine) = load_model() else {
        return;
    };
    let prompt = read_ids(&fixture("capital_france_q4k.prompt_ids"));
    let n_decode = 32usize;

    // CPU greedy trajectory.
    engine.reset_kv_for_test();
    let mut cpu_logits0 = Vec::new();
    for &t in &prompt {
        cpu_logits0 = engine.forward_token(t).expect("cpu prefill");
    }
    let mut cpu_traj = Vec::with_capacity(n_decode);
    let mut next = argmax(&cpu_logits0);
    cpu_traj.push(next);
    for _ in 1..n_decode {
        let lg = engine.forward_token(next).expect("cpu decode");
        next = argmax(&lg);
        cpu_traj.push(next);
    }

    // GPU greedy trajectory.
    engine.reset_kv_for_test();
    let mut gpu_logits0 = Vec::new();
    for &t in &prompt {
        gpu_logits0 = engine.forward_token_gpu(t).expect("gpu prefill");
    }
    let mut gpu_traj = Vec::with_capacity(n_decode);
    let mut next = argmax(&gpu_logits0);
    gpu_traj.push(next);
    for _ in 1..n_decode {
        let lg = engine.forward_token_gpu(next).expect("gpu decode");
        next = argmax(&lg);
        gpu_traj.push(next);
    }

    let matched = cpu_traj
        .iter()
        .zip(gpu_traj.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!("rwkv7 GPU greedy trajectory: {matched}/{n_decode} leading tokens match CPU oracle");
    assert_eq!(
        gpu_traj, cpu_traj,
        "GPU greedy decode must reproduce the CPU oracle trajectory for {n_decode} tokens\n  \
         cpu={cpu_traj:?}\n  gpu={gpu_traj:?}"
    );
}
