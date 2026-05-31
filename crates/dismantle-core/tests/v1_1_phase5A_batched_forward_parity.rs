//! Phase 5A parity: forward_tokens_batched (single-TCB K-token fast path) argmax
//! matches sequential forward_token calls for K=1, 2, 4, 8.
//!
//! Also verifies the exact-mode invariant: n-gram spec decode with batched verify
//! (Phase 5A) produces byte-identical output to greedy with spec off.
//!
//! Skips if model weights are not present.

use std::path::PathBuf;
use dismantle_core::{EngineConfig, GenerateRequest, SamplingParams, SpeculateMode, StreamEvent};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

fn load_engine_with_profile(speculate_mode: SpeculateMode) -> Option<Box<dyn dismantle_core::Engine>> {
    let p = weights_path();
    if !p.exists() {
        eprintln!("v1_1_phase5A_batched_forward_parity: no weights at {:?}, skipping", p);
        return None;
    }
    let mut cfg = EngineConfig::default();
    cfg.speculate = speculate_mode != SpeculateMode::Off;
    cfg.speculate_mode = speculate_mode;

    let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    if profile_path.exists() {
        if let Ok(profile) = dismantle_core::profile::KernelProfile::load(&profile_path) {
            cfg.kernel_profile = Some(profile);
        }
    }

    match dismantle_core::model::load_engine(&p, cfg) {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("v1_1_phase5A_batched_forward_parity: load failed: {err}, skipping");
            None
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn check_batched_parity(
    engine: &mut Box<dyn dismantle_core::Engine>,
    tokens: &[u32],
    positions: &[usize],
    label: &str,
) {
    // Sequential baseline.
    let seq_logits = engine
        .forward_tokens_for_test(tokens, positions)
        .unwrap_or_else(|e| panic!("{label} sequential: {e}"));

    engine.reset_kv_for_test();

    // Batched fast path (Phase 5A TCB path when conditions are met).
    let batch_logits = engine
        .forward_tokens_batched_for_test(tokens, positions)
        .unwrap_or_else(|e| panic!("{label} batched: {e}"));

    assert_eq!(seq_logits.len(), batch_logits.len(), "{label} result count mismatch");
    for m in 0..tokens.len() {
        let seq_top = argmax(&seq_logits[m]);
        let bat_top = argmax(&batch_logits[m]);
        assert_eq!(
            seq_top, bat_top,
            "{label} position {m}: batched argmax={bat_top} != sequential argmax={seq_top}"
        );
    }
    engine.reset_kv_for_test();
}

fn collect_tokens(engine: &mut Box<dyn dismantle_core::Engine>, prompt: &str, max_new_tokens: usize) -> Vec<u32> {
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
    engine.generate(req, &mut |ev| {
        if let StreamEvent::Token { id, .. } = ev {
            tokens.push(id);
        }
    }).expect("generate");
    tokens
}

/// K=1 through K=8 argmax parity: batched TCB path must match sequential.
#[test]
fn batched_tcb_argmax_parity_k1_through_k8() {
    let Some(mut engine) = load_engine_with_profile(SpeculateMode::Off) else { return };

    // K=1
    check_batched_parity(&mut engine, &[1u32], &[0], "K=1");

    // K=2
    check_batched_parity(&mut engine, &[1u32, 315], &[0, 1], "K=2");

    // K=4 (spec verify window default)
    check_batched_parity(
        &mut engine,
        &[1u32, 315, 1012, 297],
        &[0, 1, 2, 3],
        "K=4",
    );

    // K=5 (spec verify window + anchor = typical batched verify call)
    check_batched_parity(
        &mut engine,
        &[1u32, 315, 1012, 297, 338],
        &[0, 1, 2, 3, 4],
        "K=5",
    );

    // K=8
    check_batched_parity(
        &mut engine,
        &[1u32, 315, 1012, 297, 338, 263, 1243, 310],
        &[0, 1, 2, 3, 4, 5, 6, 7],
        "K=8",
    );
}

/// Exact-mode invariant with Phase 5A batched verify.
///
/// Both repetitive and natural prompts are tested with a single engine-pair load
/// to avoid GPU memory pressure from back-to-back dual-engine loads.
/// n-gram spec output must be byte-identical to greedy with spec off.
#[test]
fn spec_batched_verify_exact_mode() {
    let Some(mut ref_engine) = load_engine_with_profile(SpeculateMode::Off) else { return };
    let Some(mut spec_engine) = load_engine_with_profile(SpeculateMode::ExactShared) else { return };

    // Repetitive prompt: n-gram acceptance rate is very high.
    {
        let prompt = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let ref_ids = collect_tokens(&mut ref_engine, prompt, 16);
        let spec_ids = collect_tokens(&mut spec_engine, prompt, 16);
        assert_eq!(
            ref_ids, spec_ids,
            "repetitive: spec+batched-verify differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}"
        );
    }

    // Natural-text prompt: mixed acceptance rate.
    {
        let prompt = "Explain how speculative decoding works:";
        let ref_ids = collect_tokens(&mut ref_engine, prompt, 12);
        let spec_ids = collect_tokens(&mut spec_engine, prompt, 12);
        assert_eq!(
            ref_ids, spec_ids,
            "natural: spec+batched-verify differs from greedy\nref={ref_ids:?}\nspec={spec_ids:?}"
        );
    }
}
