#![cfg(target_os = "macos")]
use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};
mod common;
use common::argmax;
use common::weights_path_qwen as weights_path;
fn load_q4k_lmhead() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_single_tcb_tail_parity: weights missing at {w:?}");
        return None;
    }
    for v in [
        "HAWKING_QWEN_VOCAB_PRUNE",
        "HAWKING_QWEN_VOCAB_PRUNE_CORPUS",
        "HAWKING_QWEN_F16_KV",
        "HAWKING_QWEN_FLASH_ATTN",
        "HAWKING_QWEN_W4A8",
    ] {
        std::env::remove_var(v);
    }
    std::env::set_var("HAWKING_QWEN_Q4K_LMHEAD", "1"); // env_on => exact "1"; full-vocab Q4_K head
    let profile = fresh_test_profile(&w).expect("fresh test profile");
    let cfg = EngineConfig {
        kernel_profile: Some(profile),
        ..Default::default()
    };
    Some(QwenDense::load(&w, cfg).expect("load qwen-3b (q4k lm-head)"))
}
#[test]
#[ignore] // loads 1.93 GB model + sets a process-global env var; run explicitly, single-threaded
fn multiseq_single_tcb_tail_bit_identical_and_anchored() {
    let mut engine = match load_q4k_lmhead() {
        Some(e) => e,
        None => return, // weights missing — clean skip (no failure)
    };
    let vocab = engine.config.vocab_size;
    let seeds: Vec<u32> = vec![9707, 374, 100, 151643];
    let b = seeds.len();
    let n = 4usize;
    let max_seq = 16usize;
    assert!(b <= 8, "B must be <= MAX_MULTISEQ_SLOTS (8)");
    for &s in &seeds {
        engine.kv.reset();
        let single = engine
            .forward_token_greedy_tcb(s, 0)
            .expect("single-stream fwd");
        engine.multiseq_arena = None;
        let ms = engine
            .forward_tokens_multiseq(&[s], &[0], max_seq)
            .expect("B=1 multiseq fused-tail")[0];
        assert_eq!(
            single, ms,
            "anchor: B=1 fused-tail multiseq token {ms} != single-stream {single} (seed {s})"
        );
    }
    let mut solo_logits: Vec<Vec<Vec<f32>>> = Vec::with_capacity(b); // [slot][step][vocab]
    for &s in &seeds {
        engine.multiseq_arena = None;
        let mut cur = s;
        let mut steps: Vec<Vec<f32>> = Vec::with_capacity(n);
        for pos in 0..n {
            let l = engine
                .forward_tokens_multiseq_logits(&[cur], &[pos], &[0], max_seq)
                .expect("solo fused-tail logits");
            assert_eq!(l[0].len(), vocab, "solo logits must be full-vocab");
            cur = argmax(&l[0]);
            steps.push(l.into_iter().next().unwrap());
        }
        solo_logits.push(steps);
    }
    engine.multiseq_arena = None;
    let regions: Vec<usize> = (0..b).collect();
    let mut cur = seeds.clone();
    let mut batched_logits: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(n); b]; // [slot][step][vocab]
    for pos in 0..n {
        let positions = vec![pos; b];
        let l = engine
            .forward_tokens_multiseq_logits(&cur, &positions, &regions, max_seq)
            .expect("batched fused-tail logits");
        assert_eq!(l.len(), b, "batched returns B logit rows");
        let mut next = Vec::with_capacity(b);
        for (bi, row) in l.into_iter().enumerate() {
            assert_eq!(row.len(), vocab, "batched logits must be full-vocab");
            next.push(argmax(&row));
            batched_logits[bi].push(row);
        }
        cur = next;
    }
    // GATE: batched column logits BIT-IDENTICAL to solo (same fused kernel both
    for bi in 0..b {
        for step in 0..n {
            let solo = &solo_logits[bi][step];
            let batch = &batched_logits[bi][step];
            assert_eq!(
                solo.len(),
                batch.len(),
                "slot {bi} step {step}: logit length differs"
            );
            for (i, (&sv, &bv)) in solo.iter().zip(batch.iter()).enumerate() {
                assert_eq!(
                    sv.to_bits(),
                    bv.to_bits(),
                    "slot {bi} step {step} logit[{i}] NOT bit-identical (solo {sv} vs batched {bv}) \
                     — fused single-TCB tail perturbed the result (seed {})",
                    seeds[bi]
                );
            }
        }
    }
    engine.multiseq_arena = None;
    let positions0 = vec![0usize; b];
    let rerun = engine
        .forward_tokens_multiseq_logits(&seeds, &positions0, &regions, max_seq)
        .expect("rerun fused-tail logits");
    for bi in 0..b {
        let first = &batched_logits[bi][0];
        assert_eq!(
            rerun[bi].len(),
            first.len(),
            "slot {bi}: rerun length differs"
        );
        for (i, (&rv, &fv)) in rerun[bi].iter().zip(first.iter()).enumerate() {
            assert_eq!(
                rv.to_bits(),
                fv.to_bits(),
                "slot {bi} logit[{i}]: fused-tail commit not deterministic ({rv} vs {fv})"
            );
        }
    }
    let batched_tokens: Vec<Vec<u32>> = batched_logits
        .iter()
        .map(|slot| slot.iter().map(|l| argmax(l)).collect())
        .collect();
}
