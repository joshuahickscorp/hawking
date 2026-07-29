#![cfg(target_os = "macos")]
use std::path::PathBuf;
use std::sync::Mutex;
const PROMPT: &str = "The quick brown fox";
const MAX_NEW_TOKENS: usize = 16;
static ENV_LOCK: Mutex<()> = Mutex::new(());
fn find_weights() -> Option<PathBuf> {
    for candidate in ["../../models/qwen2.5-3b-instruct-q4_k_m.gguf", "models/qwen2.5-3b-instruct-q4_k_m.gguf"] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(env_path) = std::env::var("HAWKING_TEST_WEIGHTS_QWEN") {
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
fn run_greedy_capture_stats(weights: &PathBuf, cfg: hawking_core::EngineConfig) -> (Vec<u32>, hawking_core::GenStats) {
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
    let stats = engine
        .generate(req, &mut |ev| {
            if let hawking_core::StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    assert!(!ids.is_empty(), "must produce at least one token");
    (ids, stats)
}
fn clear_env() {
    std::env::remove_var("HAWKING_QWEN_TCB");
    std::env::remove_var("HAWKING_QWEN_EAGLE5");
    std::env::remove_var("HAWKING_QWEN_EAGLE5_K");
    std::env::remove_var("HAWKING_QWEN_EAGLE5_BATCHED");
    std::env::remove_var("HAWKING_QWEN_EAGLE5_CAPTURE");
    std::env::remove_var("HAWKING_QWEN_EAGLE5_CAPTURE_LAYER");
}
#[test]
fn qwen_eagle5_speculate_off_bit_identical_to_baseline() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_env();
    let Some(weights) = find_weights() else {
        eprintln!("skipping: no qwen2.5-3b-instruct-q4_k_m.gguf in models/");
        return;
    };
    let profile = find_profile(&weights);
    let cfg_a = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
    let (ids_a, _) = run_greedy_capture_stats(&weights, cfg_a);
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    let cfg_b = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
    let (ids_b, _) = run_greedy_capture_stats(&weights, cfg_b);
    clear_env();
    assert_eq!(ids_a, ids_b, "TCB-only must be bit-identical to baseline\n  baseline: {ids_a:?}\n  tcb:      {ids_b:?}");
}
#[test]
fn qwen_eagle5_speculate_on_mock_head_engages_and_preserves_greedy() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_env();
    let Some(weights) = find_weights() else {
        eprintln!("skipping: no qwen2.5-3b-instruct-q4_k_m.gguf in models/");
        return;
    };
    let profile = find_profile(&weights);
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    let cfg_baseline = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
    let (baseline_ids, _) = run_greedy_capture_stats(&weights, cfg_baseline);
    clear_env();
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::set_var("HAWKING_QWEN_EAGLE5", "1");
    std::env::set_var("HAWKING_QWEN_EAGLE5_K", "4");
    let cfg_eagle5 = hawking_core::EngineConfig {
        kernel_profile: profile.clone(),
        speculate: true,
        speculate_mode: hawking_core::SpeculateMode::Eagle5,
        eagle5_head_path: None, // forces mock head
        ..Default::default()
    };
    let (eagle5_ids, eagle5_stats) = run_greedy_capture_stats(&weights, cfg_eagle5);
    clear_env();
    assert_eq!(
        baseline_ids, eagle5_ids,
        "eagle5 spec-decode at temp=0 must emit identical tokens to no-spec greedy\n  \
         baseline: {baseline_ids:?}\n  eagle5:   {eagle5_ids:?}",
    );
    let total = eagle5_stats.draft_accepted + eagle5_stats.draft_rejected;
    assert!(
        total > 0,
        "eagle5 dispatch never engaged: draft_accepted={} draft_rejected={}",
        eagle5_stats.draft_accepted,
        eagle5_stats.draft_rejected,
    );
}
#[test]
fn qwen_eagle5_batched_mock_head_preserves_greedy() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_env();
    let Some(weights) = find_weights() else {
        eprintln!("skipping: no qwen2.5-3b-instruct-q4_k_m.gguf in models/");
        return;
    };
    let profile = find_profile(&weights);
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    let cfg_baseline = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
    let (baseline_ids, _) = run_greedy_capture_stats(&weights, cfg_baseline);
    clear_env();
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::set_var("HAWKING_QWEN_EAGLE5", "1");
    std::env::set_var("HAWKING_QWEN_EAGLE5_K", "4");
    std::env::set_var("HAWKING_QWEN_EAGLE5_BATCHED", "1");
    let cfg_batched = hawking_core::EngineConfig {
        kernel_profile: profile.clone(),
        speculate: true,
        speculate_mode: hawking_core::SpeculateMode::Eagle5,
        eagle5_head_path: None,
        ..Default::default()
    };
    let (batched_ids, batched_stats) = run_greedy_capture_stats(&weights, cfg_batched);
    clear_env();
    assert_eq!(
        baseline_ids, batched_ids,
        "batched eagle5 spec-decode at temp=0 must emit identical tokens to no-spec greedy\n  \
         baseline: {baseline_ids:?}\n  batched:  {batched_ids:?}",
    );
    let total = batched_stats.draft_accepted + batched_stats.draft_rejected;
    assert!(
        total > 0,
        "batched eagle5 dispatch never engaged: accepted={} rejected={}",
        batched_stats.draft_accepted,
        batched_stats.draft_rejected,
    );
}
#[test]
fn qwen_eagle5_capture_preserves_greedy() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_env();
    let Some(weights) = find_weights() else {
        eprintln!("skipping: no qwen2.5-3b-instruct-q4_k_m.gguf in models/");
        return;
    };
    let profile = find_profile(&weights);
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    let cfg_baseline = hawking_core::EngineConfig { kernel_profile: profile.clone(), ..Default::default() };
    let (baseline_ids, _) = run_greedy_capture_stats(&weights, cfg_baseline);
    clear_env();
    std::env::set_var("HAWKING_QWEN_TCB", "1");
    std::env::set_var("HAWKING_QWEN_EAGLE5", "1");
    std::env::set_var("HAWKING_QWEN_EAGLE5_K", "4");
    std::env::set_var("HAWKING_QWEN_EAGLE5_CAPTURE", "1");
    let cfg_capture = hawking_core::EngineConfig {
        kernel_profile: profile.clone(),
        speculate: true,
        speculate_mode: hawking_core::SpeculateMode::Eagle5,
        eagle5_head_path: None, // forces mock head (ignores capture)
        ..Default::default()
    };
    let (capture_ids, capture_stats) = run_greedy_capture_stats(&weights, cfg_capture);
    clear_env();
    assert_eq!(
        baseline_ids, capture_ids,
        "eagle5 spec-decode WITH CAPTURE at temp=0 must emit identical tokens to no-spec greedy\n  \
         baseline: {baseline_ids:?}\n  capture:  {capture_ids:?}",
    );
    let total = capture_stats.draft_accepted + capture_stats.draft_rejected;
    assert!(
        total > 0,
        "capture-mode eagle5 dispatch never engaged: accepted={} rejected={}",
        capture_stats.draft_accepted,
        capture_stats.draft_rejected,
    );
}
