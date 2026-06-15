// Phase 3.3 portability cross-check: the pure-Rust CPU reference path
// (EngineConfig.force_cpu => metal_ctx = None => forward_token + scalar dequant
// GEMV) must produce greedy output that matches the Metal path. This is the
// "engine runs correctly off-macOS" guarantee, exercised on-macOS by forcing
// the CPU path. Perf is NOT the bar (CPU decode is ~100x slower) -- correctness
// is. Scoped to the small dense qwen2.5-0.5b model (MoE CPU decode is a separate
// follow-up).

#![cfg(target_os = "macos")]

use std::path::PathBuf;

fn run_greedy(weights: &PathBuf, force_cpu: bool, n: usize) -> Vec<u32> {
    let cfg = dismantle_core::EngineConfig {
        force_cpu,
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(weights, cfg).expect("load engine");
    let req = dismantle_core::GenerateRequest {
        prompt: "The capital of France is".into(),
        max_new_tokens: n,
        sampling: dismantle_core::SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
    };
    let mut ids: Vec<u32> = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let dismantle_core::StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    ids
}

#[test]
fn cpu_backend_matches_metal_qwen05b() {
    let weights = PathBuf::from("../../models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    if !weights.exists() {
        eprintln!("skipping cpu_backend_matches_metal_qwen05b: no qwen0.5b weights");
        return;
    }
    const N: usize = 12;

    // Metal path (normal).
    let metal = run_greedy(&weights, false, N);
    // CPU reference path (force_cpu => metal_ctx = None).
    let cpu = run_greedy(&weights, true, N);

    assert!(
        metal.len() >= 3 && cpu.len() >= 3,
        "both paths must emit >=3 tokens (metal={}, cpu={})",
        metal.len(),
        cpu.len()
    );

    // Token-output parity. The CPU path dequantizes Q4_K to f32 and runs a scalar
    // gemv (f64-accumulated rmsnorm), while Metal runs the predec fused-FMA GEMV +
    // GPU argmax -- they agree only at the fp16 floor (atol~1e-3), so a LATE token
    // could diverge on a near-tie. The gate is the plan's token-parity standard:
    // the first 3 greedy IDs must match. The full match count is reported.
    let matched = metal
        .iter()
        .zip(cpu.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "CPU-vs-Metal qwen0.5b greedy: {}/{} leading tokens identical\n  metal={:?}\n  cpu  ={:?}",
        matched, N, metal, cpu
    );

    assert_eq!(
        metal[..3],
        cpu[..3],
        "first-3 greedy token IDs must match between the CPU reference path and Metal"
    );
}
