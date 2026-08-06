#![cfg(target_os = "macos")]
use hawking_core::{Engine, EngineConfig};
use std::path::PathBuf;
#[test]
fn cpu_forward_deepseek_v2_lite_force_cpu_ok() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!(
            "skipping cpu_forward_deepseek_v2_lite_force_cpu_ok: no deepseek-v2-lite-q4.gguf"
        );
        return;
    }
    let cfg = EngineConfig {
        force_cpu: true,
        ..Default::default()
    };
    let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");
    let out = engine
        .forward_tokens_for_test(&[0], &[0])
        .expect("forward_tokens_for_test (force_cpu) must return Ok");
    assert_eq!(out.len(), 1, "one input token must yield one logit vector");
    let logits = &out[0];
    assert!(
        !logits.is_empty(),
        "logits length must equal vocab_size (got {})",
        logits.len()
    );
    let bad = logits.iter().position(|v| !v.is_finite());
    assert!(
        bad.is_none(),
        "CPU-path logits must be finite (no NaN/Inf); first non-finite at index {:?} = {:?}",
        bad,
        bad.map(|i| logits[i])
    );
}
