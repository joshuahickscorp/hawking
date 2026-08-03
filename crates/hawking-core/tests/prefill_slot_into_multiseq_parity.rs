#![cfg(target_os = "macos")]
use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};
mod common;
use common::argmax;
use common::weights_path_qwen as weights_path;
fn load() -> Option<QwenDense> {
    let w = std::env::var_os("HAWKING_QWEN_PREFIX_PARITY_WEIGHTS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(weights_path);
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
    let first_token = engine
        .prefill_slot(slot_id, prompt_ids)
        .expect("prefill_slot");
    let mut cur = first_token;
    let mut seq = vec![first_token];
    for step in 1..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur], &[prompt_len + step - 1], &[slot_id])
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

fn prefix_resume_decode_after_prefill(
    engine: &mut QwenDense,
    source_slot: usize,
    target_slot: usize,
    source_prompt: &[u32],
    target_prompt: &[u32],
) -> Vec<u32> {
    use hawking_core::Engine;
    assert!(target_prompt.starts_with(source_prompt));
    engine.kv.reset();
    engine.multiseq_arena = None;
    engine.dense_arena = None;
    engine
        .prefill_slot(source_slot, source_prompt)
        .expect("materialize source prefix");
    engine
        .copy_kv_prefix_to_slot(source_slot, target_slot, source_prompt.len())
        .expect("copy exact prefix KV");
    let mut current = engine
        .prefill_slot_from_pos(target_slot, target_prompt, source_prompt.len())
        .expect("exact resumed tail");
    let mut out = vec![current];
    for step in 1..3usize {
        let logits = engine
            .forward_multiseq_batched(&[current], &[target_prompt.len() + step - 1], &[target_slot])
            .expect("resumed multiseq decode");
        current = argmax(&logits[0]);
        out.push(current);
    }
    out
}

#[test]
#[ignore]
fn copied_prefix_kv_matches_source_slot_before_any_resumed_tail() {
    use hawking_core::Engine;
    let mut engine = match load() {
        Some(engine) => engine,
        None => return,
    };
    let prompt: Vec<u32> = (1..=12).collect();
    engine
        .prefill_slot(0, &prompt)
        .expect("materialize source prefix");
    engine
        .copy_kv_prefix_to_slot(0, 3, prompt.len())
        .expect("copy exact prefix KV");
    let source_logits = engine
        .forward_multiseq_batched(&[*prompt.last().unwrap()], &[prompt.len()], &[0])
        .expect("source decode");
    let target_logits = engine
        .forward_multiseq_batched(&[*prompt.last().unwrap()], &[prompt.len()], &[3])
        .expect("copied-slot decode");
    assert_eq!(
        argmax(&target_logits[0]),
        argmax(&source_logits[0]),
        "a direct full-prefix slot copy must decode identically before tail resume"
    );
}

#[test]
#[ignore]
fn copied_prefix_then_exact_tail_matches_cold_continuation() {
    let mut engine = match load() {
        Some(engine) => engine,
        None => return,
    };
    let source: Vec<u32> = (1..=12).collect();
    let target: Vec<u32> = (1..=16).collect();
    let cold = solo_decode_after_prompt(&mut engine, &target);
    let resumed = prefix_resume_decode_after_prefill(&mut engine, 0, 3, &source, &target);
    assert_eq!(
        resumed, cold,
        "a copied exact prefix plus resumed causal tail must match cold continuation"
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
    let first_b = engine
        .prefill_slot(3, prompt_b)
        .expect("serial prefill slot 3");
    let mut cur_b = first_b;
    let mut expected_b = vec![first_b];
    for step in 1..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur_b], &[prompt_b.len() + step - 1], &[3])
            .expect("serial decode slot 3");
        let tok = argmax(&logits[0]);
        expected_b.push(tok);
        cur_b = tok;
    }
    engine.multiseq_arena = None;
    engine.kv.reset();
    engine.dense_arena = None;
    let firsts = engine
        .prefill_slots_parallel(&[(0, prompt_a), (3, prompt_b)])
        .expect("prefill_slots_parallel");
    let mut cur_a = firsts[0];
    let mut parallel_a = vec![firsts[0]];
    for step in 1..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur_a], &[prompt_a.len() + step - 1], &[0])
            .expect("parallel decode slot 0");
        let tok = argmax(&logits[0]);
        parallel_a.push(tok);
        cur_a = tok;
    }
    let mut cur_b2 = firsts[1];
    let mut parallel_b = vec![firsts[1]];
    for step in 1..3usize {
        let logits = engine
            .forward_multiseq_batched(&[cur_b2], &[prompt_b.len() + step - 1], &[3])
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
