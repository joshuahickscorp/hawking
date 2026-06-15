//! Phase 4D: n-gram spec decode exact-mode invariant.
//!
//! Greedy output with `--speculate ngram` must be byte-identical to
//! greedy output with spec off. Tests both a repetitive prompt
//! (high acceptance rate) and a natural-text prompt (mixed acceptance).
//!
//! Skips if model weights are not present.

use dismantle_core::{EngineConfig, GenerateRequest, SamplingParams, SpeculateMode, StreamEvent};
use std::path::PathBuf;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

fn load_engine(speculate_mode: SpeculateMode) -> Option<Box<dyn dismantle_core::Engine>> {
    let p = weights_path();
    if !p.exists() {
        eprintln!(
            "v1_1_phase4D_spec_exact_mode: no weights at {:?}, skipping",
            p
        );
        return None;
    }
    let mut cfg = EngineConfig::default();
    cfg.speculate = speculate_mode != SpeculateMode::Off;
    cfg.speculate_mode = speculate_mode;

    // Load profile if available — activates GPU-resident greedy path used in production.
    let profile_path = std::path::PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    if profile_path.exists() {
        if let Ok(profile) = dismantle_core::profile::KernelProfile::load(&profile_path) {
            cfg.kernel_profile = Some(profile);
        }
    }

    match dismantle_core::model::load_engine(&p, cfg) {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("v1_1_phase4D_spec_exact_mode: load failed: {err}, skipping");
            None
        }
    }
}

fn collect_tokens(
    engine: &mut Box<dyn dismantle_core::Engine>,
    prompt: &str,
    max_new_tokens: usize,
) -> Vec<u32> {
    let req = GenerateRequest {
        prompt: prompt.to_string(),
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
        },
        stop: vec![],
        abort: None,
        max_stall_ms: 0,
    };
    let mut tokens = Vec::new();
    engine
        .generate(req, &mut |ev| {
            if let StreamEvent::Token { id, .. } = ev {
                tokens.push(id);
            }
        })
        .expect("generate");
    tokens
}

/// Repetitive prompt: n-gram draft acceptance should be very high.
/// Output must be byte-identical to non-spec greedy.
#[test]
fn repetitive_prompt_spec_matches_greedy() {
    let Some(mut ref_engine) = load_engine(SpeculateMode::Off) else {
        return;
    };
    let Some(mut spec_engine) = load_engine(SpeculateMode::ExactShared) else {
        return;
    };

    let prompt =
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
    let ref_ids = collect_tokens(&mut ref_engine, prompt, 20);
    let spec_ids = collect_tokens(&mut spec_engine, prompt, 20);

    assert_eq!(
        ref_ids, spec_ids,
        "repetitive prompt: spec output differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}"
    );
}

/// Natural-text prompt: n-gram may not always match, but output must be identical.
#[test]
fn natural_prompt_spec_matches_greedy() {
    let Some(mut ref_engine) = load_engine(SpeculateMode::Off) else {
        return;
    };
    let Some(mut spec_engine) = load_engine(SpeculateMode::ExactShared) else {
        return;
    };

    let prompt = "Explain how speculative decoding works in language models:";
    let ref_ids = collect_tokens(&mut ref_engine, prompt, 15);
    let spec_ids = collect_tokens(&mut spec_engine, prompt, 15);

    assert_eq!(
        ref_ids, spec_ids,
        "natural prompt: spec output differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}"
    );
}
