//! Phase A Wedge A1 — forward_tokens_batched scaffold parity test.
//!
//! Verifies that the layer-first `forward_tokens_batched` produces the same
//! logit vectors as sequential `forward_tokens` calls at atol=1e-5.
//!
//! Skips if model weights are not present.

use std::path::PathBuf;

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

fn load_engine() -> Option<Box<dyn dismantle_core::Engine>> {
    let p = weights_path();
    if !p.exists() {
        eprintln!("v_a1_batched_scaffold_parity: no weights at {:?}, skipping", p);
        return None;
    }
    let cfg = dismantle_core::EngineConfig::default();
    match dismantle_core::model::load_engine(&p, cfg) {
        Ok(e) => Some(e),
        Err(err) => {
            eprintln!("v_a1_batched_scaffold_parity: load failed: {err}, skipping");
            None
        }
    }
}

#[test]
fn batched_scaffold_returns_n_finite_logit_vectors() {
    let Some(mut engine) = load_engine() else { return };

    let tokens = [1u32, 2, 3];
    let positions = [0usize, 1, 2];

    let results = engine
        .forward_tokens_batched_for_test(&tokens, &positions)
        .expect("forward_tokens_batched_for_test");

    assert_eq!(results.len(), tokens.len(), "must return N logit vectors");
    for (i, logits) in results.iter().enumerate() {
        assert!(!logits.is_empty(), "logits[{i}] empty");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "logits[{i}] contains non-finite values"
        );
    }
}

#[test]
fn batched_scaffold_matches_sequential_at_atol_1e5() {
    let Some(mut engine) = load_engine() else { return };

    let tokens = [1u32, 7, 42];
    let positions = [0usize, 1, 2];

    // Reference: sequential via forward_tokens.
    let seq_results = engine
        .forward_tokens_for_test(&tokens, &positions)
        .expect("forward_tokens sequential reference");

    // Reset KV so the batched pass starts from the same empty state.
    engine.reset_kv_for_test();

    // Batched (layer-first scaffold).
    let batch_results = engine
        .forward_tokens_batched_for_test(&tokens, &positions)
        .expect("forward_tokens_batched_for_test");

    assert_eq!(
        seq_results.len(),
        batch_results.len(),
        "result count mismatch"
    );
    for m in 0..tokens.len() {
        assert_eq!(
            seq_results[m].len(),
            batch_results[m].len(),
            "logit vector length mismatch at token {m}"
        );
        for (j, (&s, &b)) in seq_results[m].iter().zip(batch_results[m].iter()).enumerate() {
            let diff = (s - b).abs();
            assert!(
                diff <= 1e-5,
                "logit[{m}][{j}] diff {diff} > 1e-5 (seq={s} batch={b})"
            );
        }
    }
}

#[test]
fn batched_scaffold_empty_tokens_ok() {
    let Some(mut engine) = load_engine() else { return };

    let results = engine
        .forward_tokens_batched_for_test(&[], &[])
        .expect("empty batched forward");
    assert!(results.is_empty(), "empty input must return empty output");
}
