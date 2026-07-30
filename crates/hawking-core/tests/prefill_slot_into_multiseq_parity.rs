#![cfg(target_os = "macos")]
use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};
mod common;
use common::argmax;
use common::weights_path_qwen as weights_path;
fn load() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping prefill_slot_into_multiseq_parity: weights missing at {w:?}");
        return None;
    }
    for v in [
        "HAWKING_QWEN_VOCAB_PRUNE",
        "HAWKING_QWEN_Q4K_LMHEAD",
        "HAWKING_QWEN_F16_KV",
        "HAWKING_QWEN_FLASH_ATTN",
        "HAWKING_QWEN_W4A8",
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
fn multiseq_decode_after_prefill(
    engine: &mut QwenDense,
    slot_id: usize,
    prompt_ids: &[u32],
) -> Vec<u32> {
    let prompt_len = prompt_ids.len();
    let last_prompt_tok = *prompt_ids.last().unwrap();
    let returned = engine
        .prefill_slot(slot_id, prompt_ids)
        .expect("prefill_slot");
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
}
#[test]
#[ignore]
fn prefill_slot3_multiseq_matches_solo() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    assert_prefill_parity(&mut engine, &[1, 2, 3, 4], 3);
    assert_prefill_parity(&mut engine, &[9707, 374, 100], 3);
}
#[test]
#[ignore]
fn prefill_slots_parallel_parity() {
    use hawking_core::Engine;
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    let prompt_a: &[u32] = &[1, 2, 3, 4]; // slot 0, 4 tokens
    let prompt_b: &[u32] = &[9707, 374, 100]; // slot 3, 3 tokens (ragged)
    engine.multiseq_arena = None;
    engine.kv.reset();
    engine.dense_arena = None;
    let expected_a = multiseq_decode_after_prefill(&mut engine, 0, prompt_a);
    let returned_b = engine
        .prefill_slot(3, prompt_b)
        .expect("serial prefill slot 3");
    assert_eq!(returned_b, *prompt_b.last().unwrap());
    let mut cur_b = *prompt_b.last().unwrap();
    let mut expected_b = Vec::with_capacity(3);
    for step in 0..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur_b], &[prompt_b.len() + step], &[3])
            .expect("serial decode slot 3");
        let tok = argmax(&logits[0]);
        expected_b.push(tok);
        cur_b = tok;
    }
    engine.multiseq_arena = None;
    engine.kv.reset();
    engine.dense_arena = None;
    engine
        .prefill_slots_parallel(&[(0, prompt_a), (3, prompt_b)])
        .expect("prefill_slots_parallel");
    let mut cur_a = *prompt_a.last().unwrap();
    let mut parallel_a = Vec::with_capacity(3);
    for step in 0..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur_a], &[prompt_a.len() + step], &[0])
            .expect("parallel decode slot 0");
        let tok = argmax(&logits[0]);
        parallel_a.push(tok);
        cur_a = tok;
    }
    let mut cur_b2 = *prompt_b.last().unwrap();
    let mut parallel_b = Vec::with_capacity(3);
    for step in 0..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur_b2], &[prompt_b.len() + step], &[3])
            .expect("parallel decode slot 3");
        let tok = argmax(&logits[0]);
        parallel_b.push(tok);
        cur_b2 = tok;
    }
    assert_eq!(
        expected_a, parallel_a,
        "slot 0 diverges between serial and parallel prefill"
    );
    assert_eq!(
        expected_b, parallel_b,
        "slot 3 diverges between serial and parallel prefill"
    );
}
