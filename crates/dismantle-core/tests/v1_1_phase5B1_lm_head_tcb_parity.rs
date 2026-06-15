//! Phase 5B.1 parity: greedy argmax with the LM head folded into the global TCB
//! must be byte-identical to a second run with the same engine (determinism),
//! and the spec NGram exact-mode invariant must still hold (correctness).
//!
//! The fold path is active whenever `greedy_gpu_argmax_available()` is true and
//! the Wedge C single-TCB path is used (Off/NGram mode with profile).
//!
//! Tests:
//!   1. Two greedy runs on the same engine (KV reset between) produce identical
//!      tokens — verifies the fold path is deterministic.
//!   2. Spec NGram exact-mode invariant still holds with the fold active for
//!      both repetitive and natural-text prompts (correctness gate).
//!
//! The engine pair for test 2 is shared (single load for both prompts) to keep
//! GPU memory pressure low. Skips if model weights are not present.

use dismantle_core::{EngineConfig, GenerateRequest, SamplingParams, SpeculateMode, StreamEvent};
use std::path::PathBuf;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

fn load_engine(speculate_mode: SpeculateMode) -> Option<Box<dyn dismantle_core::Engine>> {
    let p = weights_path();
    if !p.exists() {
        eprintln!("v1_1_phase5B1: no weights at {:?}, skipping", p);
        return None;
    }
    let mut cfg = EngineConfig::default();
    cfg.speculate = speculate_mode != SpeculateMode::Off;
    cfg.speculate_mode = speculate_mode;

    // Profile enables the Wedge C TCB path and the Phase 5B.1 LM-head fold.
    let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    if profile_path.exists() {
        if let Ok(profile) = dismantle_core::profile::KernelProfile::load(&profile_path) {
            cfg.kernel_profile = Some(profile);
        }
    }

    match dismantle_core::model::load_engine(&p, cfg) {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("v1_1_phase5B1: load failed: {err}, skipping");
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

/// Two greedy runs on the SAME engine with KV reset between them must produce
/// identical tokens. This verifies the Phase 5B.1 LM-head fold is deterministic.
#[test]
fn lm_head_fold_is_deterministic() {
    let Some(mut engine) = load_engine(SpeculateMode::Off) else {
        return;
    };

    let prompts = [
        "The quick brown fox",
        "Explain how speculative decoding works:",
    ];
    for prompt in &prompts {
        engine.reset_kv_for_test();
        let run1 = collect_tokens(&mut engine, prompt, 16);

        engine.reset_kv_for_test();
        let run2 = collect_tokens(&mut engine, prompt, 16);

        assert_eq!(
            run1, run2,
            "prompt={prompt:?}: Phase 5B.1 fold not deterministic\nrun1={run1:?}\nrun2={run2:?}"
        );
        assert!(
            !run1.is_empty(),
            "prompt={prompt:?}: fold produced no tokens"
        );
    }
}

/// Spec NGram exact-mode invariant with Phase 5B.1 (LM head folded into TCB).
/// Both repetitive and natural prompts are tested with a single engine-pair load.
#[test]
fn spec_exact_mode_with_lm_head_fold() {
    let Some(mut ref_engine) = load_engine(SpeculateMode::Off) else {
        return;
    };
    let Some(mut spec_engine) = load_engine(SpeculateMode::ExactShared) else {
        return;
    };

    // Repetitive prompt (high n-gram acceptance).
    {
        let prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let ref_ids = collect_tokens(&mut ref_engine, prompt, 16);
        let spec_ids = collect_tokens(&mut spec_engine, prompt, 16);
        assert_eq!(
            ref_ids, spec_ids,
            "repetitive: spec+5B.1 differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}"
        );
    }

    // Natural-text prompt (low n-gram acceptance).
    {
        let prompt = "Explain how speculative decoding works:";
        let ref_ids = collect_tokens(&mut ref_engine, prompt, 12);
        let spec_ids = collect_tokens(&mut spec_engine, prompt, 12);
        assert_eq!(
            ref_ids, spec_ids,
            "natural: spec+5B.1 differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}"
        );
    }
}
