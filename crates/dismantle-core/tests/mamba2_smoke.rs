//! Mamba2 loader + deterministic greedy smoke.
//!
//! Auto-activates when `models/mamba2-370m-Q4_K_M.gguf` or
//! `models/mamba2-370m-f16.gguf` is present. The current engine is a
//! correctness-first CPU/Metal hybrid reference path; this test is intentionally
//! short so it can sit in the post-G1a chain as an architecture-breadth gate.

use std::path::PathBuf;

use dismantle_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};

fn locate() -> Option<PathBuf> {
    for rel in [
        "models/mamba2-370m-Q4_K_M.gguf",
        "models/mamba2-370m-f16.gguf",
    ] {
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        loop {
            let cand = dir.join(rel);
            if cand.exists() {
                return Some(cand);
            }
            if !dir.pop() {
                break;
            }
        }
    }
    None
}

fn greedy_ids(weights: &PathBuf) -> Vec<u32> {
    let mut engine =
        dismantle_core::model::load_engine(weights, EngineConfig::default()).expect("load mamba2");
    assert_eq!(engine.model_arch(), "mamba2");
    let req = GenerateRequest {
        prompt: "The capital of France is".into(),
        max_new_tokens: 4,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(0),
            ..Default::default()
        },
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    };
    let mut ids = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("mamba2 generate");
    ids
}

#[test]
fn mamba2_loads_and_greedy_is_deterministic() {
    let Some(weights) = locate() else {
        eprintln!("skipping mamba2 smoke: no mamba2 GGUF model in models/");
        return;
    };
    let a = greedy_ids(&weights);
    let b = greedy_ids(&weights);
    assert!(!a.is_empty(), "mamba2 smoke should emit at least one token");
    assert_eq!(a, b, "mamba2 temp=0 greedy decode must be deterministic");
}
