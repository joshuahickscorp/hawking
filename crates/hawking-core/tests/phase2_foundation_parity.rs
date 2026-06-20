//! Phase 2 wedges 2a/2b/2c parity. Confirms:
//!   - forward_tokens(N=3) returns 3 logit vectors of correct shape (2a)
//!   - mla_kv_append refactor leaves the integration golden hash unchanged (2b — covered indirectly by integration_greedy_64)
//!   - rope_inplace_batch produces bit-identical output to N sequential rope_inplace calls (2c)
//!
//! Skips if model not present.

use hawking_core::kernels::{rope_inplace, rope_inplace_batch};
use std::path::PathBuf;

#[test]
fn rope_batch_matches_sequential() {
    let head_dim = 64;
    let base = 10000.0_f32;

    let mut a1: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.1).collect();
    let mut a2: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.2).collect();
    let mut a3: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.3).collect();

    let mut b1 = a1.clone();
    let mut b2 = a2.clone();
    let mut b3 = a3.clone();

    // Sequential reference
    rope_inplace(&mut a1, 7, base);
    rope_inplace(&mut a2, 11, base);
    rope_inplace(&mut a3, 13, base);

    // Batch
    {
        let mut refs: Vec<&mut [f32]> = vec![&mut b1, &mut b2, &mut b3];
        rope_inplace_batch(&mut refs, &[7, 11, 13], base);
    }

    assert_eq!(a1, b1, "rope_batch[0] mismatch");
    assert_eq!(a2, b2, "rope_batch[1] mismatch");
    assert_eq!(a3, b3, "rope_batch[2] mismatch");
}

#[test]
fn rope_batch_empty_is_noop() {
    let mut empty: Vec<&mut [f32]> = vec![];
    rope_inplace_batch(&mut empty, &[], 10000.0);
    // No panic, no allocation; just confirm the empty path is safe.
}

#[test]
fn forward_tokens_shim_returns_n_vectors() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!("skipping forward_tokens_shim: no weights at {:?}", weights);
        return;
    }
    let profile_path = PathBuf::from("../../profiles/deepseek-v2-lite-q4.m3pro18.json");
    let profile =
        hawking_core::profile::KernelProfile::load(&profile_path).expect("load profile");
    let cfg = hawking_core::EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    let mut engine = hawking_core::model::load_engine(&weights, cfg).expect("load engine");

    let logits = engine
        .forward_tokens_for_test(&[1, 2, 3], &[0, 1, 2])
        .expect("forward_tokens shim");
    assert_eq!(logits.len(), 3, "must return N logit vectors for N tokens");
    for (i, lvec) in logits.iter().enumerate() {
        assert!(!lvec.is_empty(), "logits[{i}] empty");
        assert!(lvec.iter().all(|x| x.is_finite()), "logits[{i}] non-finite");
    }
}
