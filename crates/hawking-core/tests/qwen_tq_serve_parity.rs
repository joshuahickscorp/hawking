#![cfg(all(target_os = "macos", feature = "tq"))]
use hawking_core::model::load_engine;
use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};
use std::path::Path;
const WEIGHTS: &str = "../../models/Qwen2.5-3B-Instruct-Q4_K_M.gguf";
const SIDECAR: &str = "../../models/Qwen2.5-3B-Instruct-Q4_K_M.tq";
const PROMPT: &str = "The capital of France is";
const N_TOKENS: usize = 24;
fn greedy_tq_trajectory(tq_cpu: bool) -> Vec<u32> {
    std::env::set_var("HAWKING_QWEN_TQ", "1");
    std::env::remove_var("HAWKING_TQ_RESIDUAL");
    if tq_cpu {
        std::env::set_var("HAWKING_QWEN_TQ_CPU", "1");
    } else {
        std::env::remove_var("HAWKING_QWEN_TQ_CPU");
    }
    let mut engine =
        load_engine(Path::new(WEIGHTS), EngineConfig::default()).expect("load TQ engine");
    let req = GenerateRequest {
        prompt: PROMPT.to_string(),
        max_new_tokens: N_TOKENS,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(0),
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
            if let StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    ids
}
#[test]
#[ignore = "heavy: loads + greedily decodes a 3B twice; run with --ignored and the model present"]
fn qwen_tq_served_forward_is_nondegenerate_and_cpu_gpu_agree() {
    let weights = Path::new(WEIGHTS);
    let sidecar = Path::new(SIDECAR);
    if !weights.exists() || !sidecar.exists() {
        eprintln!(
            "qwen_tq_served_forward: skip (need {} + {})",
            weights.display(),
            sidecar.display()
        );
        return;
    }
    let gpu = greedy_tq_trajectory(false);
    assert!(
        !gpu.is_empty(),
        "TQ served forward produced no tokens (load or decode broke)"
    );
    let distinct = gpu.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 1,
        "TQ served forward is degenerate: {} tokens, all identical ({:?})",
        gpu.len(),
        gpu.first()
    );
    let cpu = greedy_tq_trajectory(true);
    assert_eq!(
        gpu, cpu,
        "TQ greedy trajectories diverged: GPU strand_bitslice_gemv_tcb vs CPU matvec_rht\n  gpu={gpu:?}\n  cpu={cpu:?}"
    );
}
