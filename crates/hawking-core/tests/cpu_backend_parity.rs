#![cfg(target_os = "macos")]
use std::path::PathBuf;
fn run_greedy(weights: &PathBuf, force_cpu: bool, n: usize) -> Vec<u32> {
    let cfg = hawking_core::EngineConfig {
        force_cpu,
        ..Default::default()
    };
    let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load engine");
    let req = hawking_core::GenerateRequest {
        prompt: "The capital of France is".into(),
        max_new_tokens: n,
        sampling: hawking_core::SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    };
    let mut ids: Vec<u32> = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let hawking_core::StreamEvent::Token { id, .. } = ev {
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
    let metal = run_greedy(&weights, false, N);
    let cpu = run_greedy(&weights, true, N);
    assert!(
        metal.len() >= 3 && cpu.len() >= 3,
        "both paths must emit >=3 tokens (metal={}, cpu={})",
        metal.len(),
        cpu.len()
    );
    let matched = metal
        .iter()
        .zip(cpu.iter())
        .take_while(|(a, b)| a == b)
        .count();
    assert_eq!(
        metal[..3],
        cpu[..3],
        "first-3 greedy token IDs must match between the CPU reference path and Metal"
    );
}
