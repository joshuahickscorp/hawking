#![cfg(target_os = "macos")]

use std::path::PathBuf;

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 8;

fn run_ids(max_routed_expert_ram_mb: Option<usize>) -> Vec<u32> {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    let profile =
        dismantle_core::profile::KernelProfile::load(&profile_path).expect("load profile");
    let cfg = dismantle_core::EngineConfig {
        kernel_profile: Some(profile),
        max_routed_expert_ram_mb,
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(&weights, cfg).expect("load engine");
    let req = dismantle_core::GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: dismantle_core::SamplingParams {
            temperature: 0.0,
            seed: Some(42),
            ..Default::default()
        },
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 60_000,
    };
    let mut ids = Vec::new();
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
fn v2lite_memory_limit_noop_is_bit_identical() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!("skipping memory eviction parity: no weights at {weights:?}");
        return;
    }
    let unlimited = run_ids(None);
    let aggressive = run_ids(Some(1));
    assert_eq!(aggressive, unlimited);
}
