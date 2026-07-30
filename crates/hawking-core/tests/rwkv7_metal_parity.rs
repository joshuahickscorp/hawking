#![cfg(target_os = "macos")]
use hawking_core::model::rwkv7::RwkvSeven;
use hawking_core::{Engine, EngineConfig};
use std::path::{Path, PathBuf};
mod common;
use common::*;
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
#[test]
fn rwkv7_gpu_matches_cpu_logits() {
    let Some(mut engine) = load_model() else {
        return;
    };
    let mut input = read_ids(&fixture("capital_france_q4k.prompt_ids"));
    input.extend(read_ids(&fixture("capital_france_q4k.gen_ids")));
    assert!(input.len() >= 32, "need >=32 steps, got {}", input.len());
    engine.reset_kv_for_test();
    let mut cpu_logits = Vec::with_capacity(input.len());
    for &t in &input {
        cpu_logits.push(engine.forward_token(t).expect("cpu forward"));
    }
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
        }
    }
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
#[test]
fn rwkv7_gpu_greedy_trajectory_matches_cpu() {
    let Some(mut engine) = load_model() else {
        return;
    };
    let prompt = read_ids(&fixture("capital_france_q4k.prompt_ids"));
    let n_decode = 32usize;
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
    assert_eq!(
        gpu_traj, cpu_traj,
        "GPU greedy decode must reproduce the CPU oracle trajectory for {n_decode} tokens\n  \
         cpu={cpu_traj:?}\n  gpu={gpu_traj:?}"
    );
}
