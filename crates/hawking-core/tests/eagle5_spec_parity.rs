#![cfg(target_os = "macos")]
use std::path::PathBuf;
const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 16;
fn run_greedy(weights: &PathBuf, cfg: hawking_core::EngineConfig) -> Vec<u32> {
    let mut engine = hawking_core::model::load_engine(weights, cfg).expect("load engine");
    let req = hawking_core::GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: hawking_core::SamplingParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
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
    assert!(!ids.is_empty(), "must produce at least one token");
    ids
}
fn find_weights() -> Option<PathBuf> {
    for candidate in ["../../models/deepseek-v2-lite-q4.gguf", "models/deepseek-v2-lite-q4.gguf"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(env_path) = std::env::var("HAWKING_TEST_WEIGHTS") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}
fn find_profile(weights: &PathBuf) -> Option<hawking_core::profile::KernelProfile> {
    hawking_core::profile::fresh_test_profile(weights).ok()
}
#[test]
fn eagle5_greedy_parity_k4() {
    let Some(weights) = find_weights() else {
        eprintln!("skipping eagle5_greedy_parity_k4: no deepseek-v2-lite-q4.gguf");
        return;
    };
    let profile = find_profile(&weights);
    let cfg_baseline = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
    let baseline_ids = run_greedy(&weights, cfg_baseline);
    let cfg_eagle5 = hawking_core::EngineConfig {
        kernel_profile: profile,
        speculate: true,
        speculate_mode: hawking_core::SpeculateMode::Eagle5,
        verify_window: 4,
        eagle5_head_path: None, // forces mock-head fallback
        ..Default::default()
    };
    let eagle5_ids = run_greedy(&weights, cfg_eagle5);
    assert_eq!(
        baseline_ids, eagle5_ids,
        "eagle5 spec-decode at temp=0 must emit the same tokens as no-spec greedy\n  \
         baseline: {:?}\n  eagle5:   {:?}",
        baseline_ids, eagle5_ids,
    );
}
#[test]
fn eagle5_greedy_parity_k2_and_k8() {
    let Some(weights) = find_weights() else {
        eprintln!("skipping eagle5_greedy_parity_k2_and_k8: no weights");
        return;
    };
    let profile = find_profile(&weights);
    let cfg_baseline = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
    let baseline_ids = run_greedy(&weights, cfg_baseline);
    for &k in &[2usize, 8] {
        let cfg = hawking_core::EngineConfig {
            kernel_profile: profile.clone(),
            speculate: true,
            speculate_mode: hawking_core::SpeculateMode::Eagle5,
            verify_window: k,
            eagle5_head_path: None,
            ..Default::default()
        };
        let ids = run_greedy(&weights, cfg);
        assert_eq!(
            baseline_ids, ids,
            "eagle5 K={k} must emit the same tokens as no-spec greedy\n  \
             baseline: {:?}\n  eagle5:   {:?}",
            baseline_ids, ids,
        );
    }
}
