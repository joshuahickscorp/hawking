#![cfg(target_os = "macos")]
//! Parity: `prefill_slot` correctly plants per-layer KV into the multiseq arena
//! so that subsequent `forward_multiseq_batched` decode steps produce the same
//! token sequence as a solo `forward_token_greedy_tcb` run on the same prompt.
//!
//! What this proves: the 3-step copy in `prefill_slot` —
//!   (1) batched prefill into `dense_arena` via `forward_tokens_batch_tcb`,
//!   (2) memcpy of K and V into the target slot region of `multiseq_arena`,
//!   (3) first decode step via `forward_multiseq_batched` at pos = prompt_len
//! — produces a token sequence IDENTICAL to the single-stream path. A wrong
//! `slot_base` offset, a layer stride mismatch, or a missing `dense_arena`
//! flush would all cause the multiseq decode to read stale/zero KV and diverge.
//!
//! Two sub-tests:
//!   (A) slot_id = 0 — identity case; base offset is zero, so a missing
//!       `slot_base` addition would still pass this but fail (B).
//!   (B) slot_id = 3 — non-zero slot; `slot_base = 3 * MAX_MULTISEQ_CTX * kv_dim`
//!       is required. Any wrong slot_id multiplier diverges here.
//!
//! Both are `#[ignore]` — load the 1.93 GB model. Run explicitly:
//!   cargo test --release -p dismantle-core --test prefill_slot_into_multiseq_parity \
//!     -- --ignored --test-threads=1 --nocapture

use std::path::PathBuf;

use dismantle_core::{model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

fn load() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping prefill_slot_into_multiseq_parity: weights missing at {w:?}");
        return None;
    }
    for v in [
        "DISMANTLE_QWEN_VOCAB_PRUNE",
        "DISMANTLE_QWEN_Q4K_LMHEAD",
        "DISMANTLE_QWEN_F16_KV",
        "DISMANTLE_QWEN_FLASH_ATTN",
        "DISMANTLE_QWEN_W4A8",
    ] {
        std::env::remove_var(v);
    }
    let profile = fresh_test_profile(&w).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    Some(QwenDense::load(&w, cfg).expect("load qwen-3b"))
}

fn argmax(l: &[f32]) -> u32 {
    l.iter()
        .enumerate()
        .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv { (i as u32, v) } else { (bi, bv) }
        })
        .0
}

/// Decode 3 tokens via the solo single-stream path starting after the prompt.
/// Runs `forward_token_greedy_tcb` for each prompt token, then 3 more decode
/// steps. Returns the 3 greedy output token IDs.
fn solo_decode_after_prompt(engine: &mut QwenDense, prompt_ids: &[u32]) -> Vec<u32> {
    engine.kv.reset();
    engine.multiseq_arena = None;
    engine.dense_arena = None;

    let prompt_len = prompt_ids.len();
    assert!(!prompt_ids.is_empty());

    let mut last_out = 0u32;
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        last_out = engine
            .forward_token_greedy_tcb(tok, pos)
            .expect("solo prefill step");
    }
    let mut seq = vec![last_out];
    let mut cur = last_out;
    for step in 1..3usize {
        cur = engine
            .forward_token_greedy_tcb(cur, prompt_len + step - 1)
            .expect("solo decode step");
        seq.push(cur);
    }
    seq
}

/// Decode 3 tokens via the multiseq path after `prefill_slot(slot_id, prompt_ids)`.
fn multiseq_decode_after_prefill(
    engine: &mut QwenDense,
    slot_id: usize,
    prompt_ids: &[u32],
) -> Vec<u32> {
    let prompt_len = prompt_ids.len();
    let last_prompt_tok = *prompt_ids.last().unwrap();

    let returned = engine.prefill_slot(slot_id, prompt_ids).expect("prefill_slot");
    assert_eq!(
        returned, last_prompt_tok,
        "prefill_slot must return the last prompt token id"
    );

    let mut cur = last_prompt_tok;
    let mut seq = Vec::with_capacity(3);
    for step in 0..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur], &[prompt_len + step], &[slot_id])
            .expect("multiseq decode step");
        let tok = argmax(&logits[0]);
        seq.push(tok);
        cur = tok;
    }
    seq
}

fn assert_prefill_parity(engine: &mut QwenDense, prompt_ids: &[u32], slot_id: usize) {
    let solo = solo_decode_after_prompt(engine, prompt_ids);

    engine.multiseq_arena = None;
    engine.kv.reset();
    engine.dense_arena = None;

    let multi = multiseq_decode_after_prefill(engine, slot_id, prompt_ids);

    println!(
        "[prefill-slot-parity] slot_id={slot_id} prompt={prompt_ids:?} solo={solo:?} multi={multi:?}"
    );
    assert_eq!(
        solo, multi,
        "prefill_slot KV copy is wrong — multiseq token diverges from solo \
         (slot_id={slot_id}, prompt={prompt_ids:?})"
    );
}

#[test]
#[ignore]
fn prefill_slot0_multiseq_matches_solo() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    assert_prefill_parity(&mut engine, &[1, 2, 3, 4], 0);
    assert_prefill_parity(&mut engine, &[9707, 374, 100], 0);
    println!("[prefill-slot-parity] slot_id=0: PASS");
}

/// Non-zero slot — `slot_base = 3 * MAX_MULTISEQ_CTX * kv_dim` is required.
/// A missing or wrong slot_id multiplier diverges here while slot_id=0 passes.
#[test]
#[ignore]
fn prefill_slot3_multiseq_matches_solo() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    assert_prefill_parity(&mut engine, &[1, 2, 3, 4], 3);
    assert_prefill_parity(&mut engine, &[9707, 374, 100], 3);
    println!("[prefill-slot-parity] slot_id=3: PASS");
}
