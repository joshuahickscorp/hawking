#![cfg(target_os = "macos")]
//! Continuous-batching SLOT-CHURN parity — the test that catches the
//! arena/slot-id bug the review found (and that the lockstep equivalence test
//! structurally cannot).
//!
//! Multi-seq KV is keyed by a STABLE region (the slot id), not the compacted
//! dispatch index. So when the active/ready set SHRINKS (a slot hits EOS and is
//! evicted), the surviving slots must keep reading their OWN KV history — even
//! though their compacted batch index shifted. With the pre-fix code (KV keyed
//! by compacted index), an evicted slot's neighbour would read the EVICTED
//! slot's KV → silent cross-sequence contamination. This test reproduces an
//! eviction and asserts each survivor's tokens still equal its solo decode.
//!
//! Skipped if weights are missing.

use std::path::PathBuf;

use dismantle_core::{model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

fn load() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_churn_parity: weights missing at {w:?}");
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
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

/// One multi-seq step → per-slot argmax token (via the logits seam + explicit regions).
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

    // Solo refs: each seed decoded ALONE (region 0, fresh KV) for 3 steps.
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

    // Churned run: regions stay STABLE [0,1,2]; slot 1 is EVICTED before step 2.
    engine.multiseq_arena = None;
    // step 0 — all three active at pos 0, stable regions [0,1,2].
    let s0 = step(&mut engine, &seeds, &[0, 0, 0], &[0, 1, 2], max_seq);
    // step 1 — all three at pos 1.
    let s1 = step(&mut engine, &s0, &[1, 1, 1], &[0, 1, 2], max_seq);
    // step 2 — slot 1 EVICTED. Active = [slot0, slot2], COMPACTED to batch [0,1]
    // but with stable regions [0,2] and positions [2,2]. Pre-fix, slot2 (compacted
    // index 1) would read region 1 = the evicted slot1's KV.
    let s2 = step(&mut engine, &[s1[0], s1[2]], &[2, 2], &[0, 2], max_seq);

    let churn_slot0 = vec![s0[0], s1[0], s2[0]]; // region 0, compacted idx 0 throughout
    let churn_slot2 = vec![s0[2], s1[2], s2[1]]; // region 2, compacted idx 2 then 1 after evict

    println!("[multiseq-churn] solo0={:?} churn0={churn_slot0:?}", solo[0]);
    println!("[multiseq-churn] solo2={:?} churn2={churn_slot2:?}", solo[2]);
    assert_eq!(solo[0], churn_slot0, "slot0 churned tokens != solo decode");
    assert_eq!(
        solo[2], churn_slot2,
        "slot2 (after slot1 eviction + index compaction) != solo decode — KV CROSS-CONTAMINATION"
    );
}
