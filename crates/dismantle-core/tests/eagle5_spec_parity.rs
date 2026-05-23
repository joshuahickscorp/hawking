// Eagle5 spec-decode greedy parity test.
//
// The spec-decode correctness invariant is that greedy generation at
// temperature=0 produces the same token sequence whether or not
// speculative decoding is enabled, regardless of the draft head's
// accept rate. The draft only proposes; the verifier (the full
// V2-Lite model) takes the final argmax at every position.
//
// This test exercises that invariant against the deterministic mock
// Eagle5 head — its accept rate is near 1/vocab, so most steps will
// emit only the verifier's bonus token, but the emitted sequence
// must match exactly the no-spec greedy run. If this test ever fails
// it means the spec-decode runtime is changing greedy output, which
// is a correctness bug (not a perf regression).
//
// Mac-only because the engine constructor needs the Metal context.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

const PROMPT: &str = "Once upon a time";
const MAX_NEW_TOKENS: usize = 16;

fn run_greedy(
    weights: &PathBuf,
    cfg: dismantle_core::EngineConfig,
) -> Vec<u32> {
    let mut engine =
        dismantle_core::model::load_engine(weights, cfg).expect("load engine");
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
    engine
        .generate(req, &mut |ev| {
            if let dismantle_core::StreamEvent::Token { id, .. } = ev {
                ids.push(id);
            }
        })
        .expect("generate");
    assert!(!ids.is_empty(), "must produce at least one token");
    ids
}

fn find_weights() -> Option<PathBuf> {
    // Test cwd is the crate root. The worktree may not have a models/
    // tree; the main checkout usually does. Probe both relative roots
    // and an env-var override (used by CI).
    for candidate in [
        "../../models/deepseek-v2-lite-q4.gguf",
        "models/deepseek-v2-lite-q4.gguf",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    if let Ok(env_path) = std::env::var("DISMANTLE_TEST_WEIGHTS") {
        let p = PathBuf::from(env_path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn find_profile() -> Option<dismantle_core::profile::KernelProfile> {
    for candidate in [
        "../../profiles/deepseek-v2-lite-q4.m3pro18.json",
        "profiles/deepseek-v2-lite-q4.m3pro18.json",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            return dismantle_core::profile::KernelProfile::load(&p).ok();
        }
    }
    None
}

#[test]
fn eagle5_greedy_parity_k4() {
    let Some(weights) = find_weights() else {
        eprintln!("skipping eagle5_greedy_parity_k4: no deepseek-v2-lite-q4.gguf");
        return;
    };
    let profile = find_profile();

    // Baseline: no-spec greedy.
    let cfg_baseline = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        ..Default::default()
    };
    let baseline_ids = run_greedy(&weights, cfg_baseline);

    // Eagle5 spec-decode greedy with mock head + K=4.
    let cfg_eagle5 = dismantle_core::EngineConfig {
        kernel_profile: profile,
        speculate: true,
        speculate_mode: dismantle_core::SpeculateMode::Eagle5,
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
    // Same invariant at the other supported window sizes — sanity check
    // that the verify path's accept-prefix accounting is correct across
    // K∈{2,8} too.
    let Some(weights) = find_weights() else {
        eprintln!("skipping eagle5_greedy_parity_k2_and_k8: no weights");
        return;
    };
    let profile = find_profile();

    let cfg_baseline = dismantle_core::EngineConfig {
        kernel_profile: profile.clone(),
        ..Default::default()
    };
    let baseline_ids = run_greedy(&weights, cfg_baseline);

    for &k in &[2usize, 8] {
        let cfg = dismantle_core::EngineConfig {
            kernel_profile: profile.clone(),
            speculate: true,
            speculate_mode: dismantle_core::SpeculateMode::Eagle5,
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
