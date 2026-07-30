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
        eprintln!("skipping multiseq_churn_parity: weights missing at {w:?}");
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
fn step(
    engine: &mut QwenDense,
    tokens: &[u32],
    positions: &[usize],
    regions: &[usize],
    max_seq: usize,
) -> Vec<u32> {
    engine
        .forward_tokens_multiseq_logits(tokens, positions, regions, max_seq)
        .expect("multiseq logits")
        .iter()
        .map(|l| argmax(l))
        .collect()
}
#[test]
fn multiseq_survives_slot_eviction() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    let max_seq = 16usize;
    let seeds = [9707u32, 374, 100];
    let mut solo: Vec<Vec<u32>> = Vec::new();
    for &s in &seeds {
        engine.multiseq_arena = None;
        let mut cur = s;
        let mut seq = Vec::new();
        for pos in 0..3usize {
            let next = step(&mut engine, &[cur], &[pos], &[0], max_seq);
            seq.push(next[0]);
            cur = next[0];
        }
        solo.push(seq);
    }
    engine.multiseq_arena = None;
    let s0 = step(&mut engine, &seeds, &[0, 0, 0], &[0, 1, 2], max_seq);
    let s1 = step(&mut engine, &s0, &[1, 1, 1], &[0, 1, 2], max_seq);
    let s2 = step(&mut engine, &[s1[0], s1[2]], &[2, 2], &[0, 2], max_seq);
    let churn_slot0 = vec![s0[0], s1[0], s2[0]]; // region 0, compacted idx 0 throughout
    let churn_slot2 = vec![s0[2], s1[2], s2[1]]; // region 2, compacted idx 2 then 1 after evict
    assert_eq!(solo[0], churn_slot0, "slot0 churned tokens != solo decode");
    assert_eq!(
        solo[2], churn_slot2,
        "slot2 (after slot1 eviction + index compaction) != solo decode — KV CROSS-CONTAMINATION"
    );
}
#[test]
fn multiseq_arena_no_realloc_on_ctx_change() {
    let mut engine = match load() {
        Some(e) => e,
        None => return,
    };
    let seed = 9707u32;
    let region = 0usize;
    let small = 16usize;
    let large = 2048usize;
    engine.multiseq_arena = None;
    let solo_s0 = engine
        .forward_tokens_multiseq_logits(&[seed], &[0], &[region], large)
        .expect("solo s0");
    let solo_tok0 = argmax(&solo_s0[0]);
    let solo_s1 = engine
        .forward_tokens_multiseq_logits(&[solo_tok0], &[1], &[region], large)
        .expect("solo s1");
    let solo_tok1 = argmax(&solo_s1[0]);
    engine.multiseq_arena = None;
    let trig_s0 = engine
        .forward_tokens_multiseq_logits(&[seed], &[0], &[region], small)
        .expect("trigger s0 (small)");
    let trig_tok0 = argmax(&trig_s0[0]);
    let trig_s1 = engine
        .forward_tokens_multiseq_logits(&[trig_tok0], &[1], &[region], large)
        .expect("trigger s1 (large)");
    let trig_tok1 = argmax(&trig_s1[0]);
    assert_eq!(
        solo_tok0, trig_tok0,
        "step-0 token differs — KV layout is max_seq_per_slot-dependent at step 0"
    );
    assert_eq!(
        solo_tok1, trig_tok1,
        "arena reallocated on max_seq_per_slot change — slot KV wiped: \
         pre-fix realloc bug reproduced (solo={solo_tok1}, trigger={trig_tok1})"
    );
}
