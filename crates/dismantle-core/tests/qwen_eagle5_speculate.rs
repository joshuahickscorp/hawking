//! Qwen Eagle5 spec-decode tests (Phase B port).
//!
//! Two invariants this file guards:
//!
//! 1. **Speculate=off equivalence:** running with `DISMANTLE_QWEN_TCB=1` and
//!    no Eagle5 must produce bit-identical greedy tokens to baseline. Eagle5
//!    must never silently perturb the default path.
//! 2. **Speculate=eagle5 engagement:** with `DISMANTLE_QWEN_EAGLE5=1` and
//!    `speculate_mode==Eagle5` (mock head), the spec-decode branch engages,
//!    `draft_accepted + draft_rejected > 0`, AND emitted tokens still match
//!    the no-spec greedy run (correctness preservation at temp=0).
//!
//! Mock head is used (no `eagle5_head_path` set) so the test doesn't need
//! a 1.66 GB trained-head safetensors on disk. Accept rate of the mock
//! head is near 1/vocab; what we're proving is the dispatch path works
//! end-to-end without breaking greedy output.
//!
//! Mac-only because the engine constructor needs a Metal context.

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::sync::Mutex;

const PROMPT: &str = "The quick brown fox";
const MAX_NEW_TOKENS: usize = 16;

// Env flags `DISMANTLE_QWEN_TCB` / `DISMANTLE_QWEN_EAGLE5` are process-global;
// cargo test runs all tests in one binary, so we serialize the env-mutating
// tests in this file to avoid interleaving.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn find_weights() -> Option<PathBuf> {
    for candidate in [
        "../../models/qwen2.5-3b-instruct-q4_k_m.gguf",
        "models/qwen2.5-3b-instruct-q4_k_m.gguf",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(env_path) = std::env::var("DISMANTLE_TEST_WEIGHTS_QWEN") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_profile(weights: &PathBuf) -> Option<dismantle_core::profile::KernelProfile> {
    dismantle_core::profile::fresh_test_profile(weights).ok()
}

fn run_greedy_capture_stats(
    weights: &PathBuf,
    cfg: dismantle_core::EngineConfig,
) -> (Vec<u32>, dismantle_core::GenStats) {
    let mut engine = dismantle_core::model::load_engine(weights, cfg).expect("load engine");
    let req = dismantle_core::GenerateRequest {
        prompt: PROMPT.into(),
        max_new_tokens: MAX_NEW_TOKENS,
        sampling: dismantle_core::SamplingParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
    };
    let mut ids: Vec<u32> = Vec::new();
    let stats = engine
        .generate(req, &mut |ev| {
            if let dismantle_core::StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    assert!(!ids.is_empty(), "must produce at least one token");
    (ids, stats)
}

/// Reset all Eagle5/TCB env flags so subsequent code observes a clean
/// state. Called by every test under the ENV_LOCK.
fn clear_env() {
    std::env::remove_var("DISMANTLE_QWEN_TCB");
    std::env::remove_var("DISMANTLE_QWEN_EAGLE5");
    std::env::remove_var("DISMANTLE_QWEN_EAGLE5_K");
    std::env::remove_var("DISMANTLE_QWEN_EAGLE5_BATCHED");
    std::env::remove_var("DISMANTLE_QWEN_EAGLE5_CAPTURE");
    std::env::remove_var("DISMANTLE_QWEN_EAGLE5_CAPTURE_LAYER");
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

    // Baseline 1: no env flags, no spec.
    let cfg_a = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        ..Default::default()
    };
    let (ids_a, _) = run_greedy_capture_stats(&weights, cfg_a);

    // Baseline 2: TCB on but no Eagle5. Should be identical.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    let cfg_b = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        ..Default::default()
    };
    let (ids_b, _) = run_greedy_capture_stats(&weights, cfg_b);
    clear_env();

    assert_eq!(
        ids_a, ids_b,
        "TCB-only must be bit-identical to baseline\n  baseline: {ids_a:?}\n  tcb:      {ids_b:?}"
    );
    eprintln!(
        "speculate=off path is stable across TCB toggle ({} tokens)",
        ids_a.len()
    );
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

    // Baseline: no spec-decode (TCB on for fair compare).
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    let cfg_baseline = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        ..Default::default()
    };
    let (baseline_ids, _) = run_greedy_capture_stats(&weights, cfg_baseline);
    clear_env();

    // Eagle5 spec-decode with mock head, K=4.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5", "1");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5_K", "4");
    let cfg_eagle5 = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        speculate: true,
        speculate_mode: dismantle_core::SpeculateMode::Eagle5,
        eagle5_head_path: None, // forces mock head
        ..Default::default()
    };
    let (eagle5_ids, eagle5_stats) = run_greedy_capture_stats(&weights, cfg_eagle5);
    clear_env();

    // Correctness preservation: spec-decode greedy at temp=0 must emit
    // the same tokens as no-spec greedy regardless of draft accept rate.
    assert_eq!(
        baseline_ids, eagle5_ids,
        "eagle5 spec-decode at temp=0 must emit identical tokens to no-spec greedy\n  \
         baseline: {baseline_ids:?}\n  eagle5:   {eagle5_ids:?}",
    );

    // Engagement: the dispatch must have actually run (draft counters > 0).
    // Mock head has ~1/vocab acceptance, so we expect mostly rejections,
    // but the sum must be positive — proving the spec-decode loop ran.
    let total = eagle5_stats.draft_accepted + eagle5_stats.draft_rejected;
    assert!(
        total > 0,
        "eagle5 dispatch never engaged: draft_accepted={} draft_rejected={}",
        eagle5_stats.draft_accepted,
        eagle5_stats.draft_rejected,
    );
    eprintln!(
        "eagle5 dispatch engaged: accepted={} rejected={} ({} tokens emitted)",
        eagle5_stats.draft_accepted,
        eagle5_stats.draft_rejected,
        eagle5_ids.len()
    );
}

/// Phase B.5.4 — batched verify must preserve greedy parity.
///
/// Same invariant as the serial test, but with
/// `DISMANTLE_QWEN_EAGLE5_BATCHED=1` flipped on. The batched-with-logits
/// dispatch must produce identical tokens AND increment counters.
/// Without this gate the entire spec-decode-with-batched-verify path
/// could silently emit wrong tokens.
#[test]
fn qwen_eagle5_batched_mock_head_preserves_greedy() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_env();
    let Some(weights) = find_weights() else {
        eprintln!("skipping: no qwen2.5-3b-instruct-q4_k_m.gguf in models/");
        return;
    };
    let profile = find_profile(&weights);

    // Baseline.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    let cfg_baseline = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        ..Default::default()
    };
    let (baseline_ids, _) = run_greedy_capture_stats(&weights, cfg_baseline);
    clear_env();

    // Eagle5 + batched verify with mock head.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5", "1");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5_K", "4");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5_BATCHED", "1");
    let cfg_batched = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        speculate: true,
        speculate_mode: dismantle_core::SpeculateMode::Eagle5,
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
    eprintln!(
        "batched eagle5 engaged: accepted={} rejected={} ({} tokens emitted)",
        batched_stats.draft_accepted,
        batched_stats.draft_rejected,
        batched_ids.len()
    );
}

/// Phase B.3.3 — capture-layer plumbing must preserve greedy parity.
///
/// With `DISMANTLE_QWEN_EAGLE5_CAPTURE=1` the per-layer forward inserts
/// two `memcpy_f32_off_tcb` dispatches at the chosen layer to copy
/// residual + intermediate streams into sidecar buffers. The dispatches
/// are pure copies — they do not perturb the values used by subsequent
/// layers — so greedy output MUST be bit-identical to baseline.
///
/// Mock head ignores capture (Mock has no in_proj), so accept rate
/// won't change — what we're verifying is the capture dispatches don't
/// silently corrupt the verifier's forward.
#[test]
fn qwen_eagle5_capture_preserves_greedy() {
    let _g = ENV_LOCK.lock().unwrap();
    clear_env();
    let Some(weights) = find_weights() else {
        eprintln!("skipping: no qwen2.5-3b-instruct-q4_k_m.gguf in models/");
        return;
    };
    let profile = find_profile(&weights);

    // Baseline.
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    let cfg_baseline = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        ..Default::default()
    };
    let (baseline_ids, _) = run_greedy_capture_stats(&weights, cfg_baseline);
    clear_env();

    // Eagle5 + capture (serial verify, captures populate after first cycle).
    std::env::set_var("DISMANTLE_QWEN_TCB", "1");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5", "1");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5_K", "4");
    std::env::set_var("DISMANTLE_QWEN_EAGLE5_CAPTURE", "1");
    let cfg_capture = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        speculate: true,
        speculate_mode: dismantle_core::SpeculateMode::Eagle5,
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
    eprintln!(
        "capture-mode eagle5 engaged: accepted={} rejected={} ({} tokens emitted)",
        capture_stats.draft_accepted,
        capture_stats.draft_rejected,
        capture_ids.len()
    );
}
