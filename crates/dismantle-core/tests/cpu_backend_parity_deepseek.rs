// Phase 3.3 portability LIGHT gate (MoE): the pure-Rust CPU reference path
// (EngineConfig.force_cpu => metal_ctx = None => forward_token + materialized-KV
// attention + per-routed-expert dequant-Q4_K GEMV) must LOAD and run a single
// CPU forward for the DeepSeek-V2-Lite MoE model, producing finite logits of
// length vocab_size. This is the MoE analogue of cpu_backend_parity.rs (which
// covers the dense qwen0.5b model), but intentionally a LIGHT gate: ONE 10GB
// load + ONE CPU token, NOT a CPU-vs-Metal double-load greedy comparison. Perf
// is NOT the bar (CPU MoE decode is ~100x slower) -- reach/correctness is.
//
// Unblocked by three load/forward gating fixes in deepseek_v2.rs (applied A->B->C):
//   A. load() now honors force_cpu (metal_ctx = None), mirroring qwen_dense.rs.
//   B. mla_metal is suppressed under force_cpu / off-macOS so mla_c_kv stays
//      empty and attention() takes the CPU materialized-KV path instead of the
//      'mla_decode: Metal context unavailable' hard error.
//   C. forward_token_final_norm_maybe_read gained a pure-Rust per-layer CPU
//      driver (calling the full ffn(), routed + shared experts) when
//      metal_ctx.is_none().
//
// NOTE: EngineConfig carries NO kernel_profile here (Default), so the default
// mla_metal=true is exercised at load and MUST be suppressed by Edit B for this
// to reach the FFN instead of erroring. forward_token(token, pos) is a private
// inherent method; the public trait seam is forward_tokens_for_test, which for
// deepseek routes through forward_tokens -> forward_token -> ... -> the Edit C
// CPU loop -> gemv_f16 LM head producing vec![0.0; vocab_size].
//
// RUN (on the dev Mac; this sandbox cannot dispatch):
//   cargo test -p dismantle-core --test cpu_backend_parity_deepseek -- --nocapture

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use dismantle_core::{Engine, EngineConfig};

#[test]
fn cpu_forward_deepseek_v2_lite_force_cpu_ok() {
    let weights = PathBuf::from("../../models/deepseek-v2-lite-q4.gguf");
    if !weights.exists() {
        eprintln!(
            "skipping cpu_forward_deepseek_v2_lite_force_cpu_ok: no deepseek-v2-lite-q4.gguf"
        );
        return;
    }

    // force_cpu=true => metal_ctx = None (Edit A) => mla_metal suppressed (Edit B)
    // => single CPU forward via the Edit C per-layer driver.
    let cfg = EngineConfig {
        force_cpu: true,
        ..Default::default()
    };
    let mut engine = dismantle_core::model::load_engine(&weights, cfg).expect("load engine");

    // Single CPU forward at (token=0, pos=0). forward_tokens_for_test is the
    // public trait seam; for deepseek it loops forward_token, exercising the
    // CPU decode path end-to-end (materialized-KV attention + MoE expert GEMVs).
    let out = engine
        .forward_tokens_for_test(&[0], &[0])
        .expect("forward_tokens_for_test (force_cpu) must return Ok");

    assert_eq!(out.len(), 1, "one input token must yield one logit vector");
    let logits = &out[0];

    // logits is built as vec![0.0f32; config.vocab_size] by the LM head, so its
    // length IS the model's vocab_size. Assert it is the full (non-empty) vocab
    // (Default config has no vocab-prune path, so no shrink).
    assert!(
        !logits.is_empty(),
        "logits length must equal vocab_size (got {})",
        logits.len()
    );

    // The real correctness gate: every logit must be finite (no NaN / Inf). A
    // NaN here would mean the CPU attention / MoE expert path produced garbage.
    let bad = logits.iter().position(|v| !v.is_finite());
    assert!(
        bad.is_none(),
        "CPU-path logits must be finite (no NaN/Inf); first non-finite at index {:?} = {:?}",
        bad,
        bad.map(|i| logits[i])
    );

    eprintln!(
        "deepseek-v2-lite force_cpu single CPU forward OK: vocab_size={}, all logits finite",
        logits.len()
    );
}

