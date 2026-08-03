#![cfg(target_os = "macos")]
use std::path::PathBuf;
#[test]
fn forward_token_shared_only_returns_finite_logits() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!("skipping v0511 smoke: no weights at {:?}", weights);
        return;
    }
    let profile_path = PathBuf::from(
        "../../workspace/campaign/config/profiles/deepseek-v2-lite/baseline/deepseek-v2-lite-q4.m3pro18.json",
    );
    let profile = hawking_core::profile::KernelProfile::load(&profile_path).expect("load profile");
    let cfg = hawking_core::EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");
    let token = 1u32;
    let pos = 0usize;
    let logits = engine
        .forward_token_shared_only_for_test(token, pos)
        .expect("forward_token_shared_only_for_test");
    assert!(logits.len() > 1000, "logits too short: {}", logits.len());
    let non_finite: Vec<usize> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_finite())
        .map(|(i, _)| i)
        .collect();
    assert!(
        non_finite.is_empty(),
        "non-finite logits at indices: {:?}",
        &non_finite[..non_finite.len().min(5)]
    );
}
