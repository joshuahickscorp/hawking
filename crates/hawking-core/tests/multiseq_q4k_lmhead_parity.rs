#![cfg(target_os = "macos")]
//! R1 parity: GPU-batched Q4_K LM head over multi-seq slots == per-slot SOLO.
//!
//! Under HAWKING_QWEN_Q4K_LMHEAD=1, `forward_tokens_multiseq_logits` runs the
//! LM head as ONE batched v3w Q4_K GEMM across B decode slots (one weight read,
//! broadcast over B columns) instead of B sequential CPU full-vocab matmuls. The
//! output stays FULL-vocab (length = config.vocab_size), so a column's argmax
//! index IS a real token id (no prune/remap). This test proves the batched path
//! does not contaminate slots: each batched column's argmax token equals that
//! slot decoded ALONE (B=1) through the IDENTICAL flagged path.
//!
//! Why this is NOT a tautology: the SOLO reference runs B=1 (the v3w kernel reads
//! the weight once per single column), while the BATCHED run packs B columns into
//! ONE dispatch that reads the weight once and broadcasts it across all B via the
//! kernel's two-float4 accumulators. A broadcast/staging bug (wrong x_tile slot,
//! wrong column stride, cross-slot accumulation) makes a batched column diverge
//! from its solo decode — that is exactly what the assert_eq! catches. Both sides
//! go through the SAME LM-head code under the SAME flag, so there is no
//! GPU-vs-CPU or prune-vs-full asymmetry to mask a real bug.
//!
//! Both tests set HAWKING_QWEN_Q4K_LMHEAD=1 BEFORE load (the full-vocab Q4_K
//! head is built at load time, qwen_dense.rs:878), so they must run in their OWN
//! test binary, single-threaded — set_var/remove_var is process-global and the
//! sibling multiseq tests remove_var the same key. `#[ignore]` because they load
//! the 1.93 GB Qwen-3B model. Skipped (clean) if weights are missing.
//!
//! Run (do NOT run while a CPU-heavy job holds the machine):
//!   cargo test --release -p hawking-core --test multiseq_q4k_lmhead_parity \
//!     -- --ignored --test-threads=1 --nocapture

use std::path::PathBuf;

use hawking_core::{
    model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig,
};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

/// Load Qwen-3B with the GPU Q4_K LM head FORCED ON. env_on() requires the value
/// to be exactly "1", and the Q4_K LM-head buffer is built at LOAD time, so the
/// var MUST be set before QwenDense::load. We pin every OTHER lever to its
/// default so the only variable under test is the batched Q4_K LM head.
fn load_q4k_lmhead() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_q4k_lmhead_parity: weights missing at {w:?}");
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

/// Decode `n` steps of ONE seed ALONE via the multiseq path (B=1, fresh KV),
/// under the same flagged Q4_K LM head. Returns per-step argmax tokens.
fn solo_tokens(engine: &mut QwenDense, seed: u32, n: usize, max_seq: usize) -> Vec<u32> {
    engine.multiseq_arena = None;
    let mut cur = seed;
    let mut out = Vec::with_capacity(n);
    for pos in 0..n {
        let next = engine
            .forward_tokens_multiseq(&[cur], &[pos], max_seq)
            .expect("solo q4k lm-head");
        out.push(next[0]);
        cur = next[0];
    }
    out
}

#[test]
#[ignore] // loads 1.93 GB model + sets a process-global env var; run explicitly, single-threaded
fn multiseq_q4k_lmhead_batched_equals_solo() {
    let mut engine = match load_q4k_lmhead() {
        Some(e) => e,
        None => return, // weights missing — clean skip (no failure)
    };

    // B distinct seeds (slots diverge in content), lockstep step count. B <= 8.
    let seeds: Vec<u32> = vec![9707, 374, 100, 151643];
    let b = seeds.len();
    let n = 4usize;
    let max_seq = 16usize;
    assert!(b <= 8, "B must be <= MAX_MULTISEQ_SLOTS (8)");

    // SOLO references: each seed decoded ALONE under the flagged Q4_K LM head.
    let solo: Vec<Vec<u32>> = seeds
        .iter()
        .map(|&s| solo_tokens(&mut engine, s, n, max_seq))
        .collect();

    // BATCHED: all B seeds decoded TOGETHER -> ONE Q4_K GEMM over B columns.
    engine.multiseq_arena = None;
    let mut cur = seeds.clone();
    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); b];
    for pos in 0..n {
        let positions = vec![pos; b];
        let next = engine
            .forward_tokens_multiseq(&cur, &positions, max_seq)
            .expect("batched q4k lm-head");
        for bi in 0..b {
            batched[bi].push(next[bi]);
        }
        cur = next;
    }

    // GATE: per-slot argmax token byte-identical (batched column == solo).
    for bi in 0..b {
        assert_eq!(
            solo[bi], batched[bi],
            "slot {bi}: batched Q4_K LM-head argmax {:?} != solo {:?} (seed {})",
            batched[bi], solo[bi], seeds[bi]
        );
    }
    println!("[multiseq-q4k-lmhead] B={b} batched-argmax == solo over {n} steps: {batched:?}");
}

/// Divergent-POSITION variant: B slots at DISTINCT positions (incl. a
/// long-context slot), explicit stable regions [0..b). Catches per-slot indexing
/// bugs in the batched Q4_K LM head that lockstep cannot. Uses the *_logits seam
/// directly so regions are explicit; argmax is over the FULL-vocab vector.
#[test]
#[ignore]
fn multiseq_q4k_lmhead_divergent_positions_equal_solo() {
    let mut engine = match load_q4k_lmhead() {
        Some(e) => e,
        None => return,
    };
    let max_seq = 2048usize; // accommodate the long-context slot
    let seeds = [9707u32, 374, 100, 151643];
    let start_pos = [2047usize, 1024, 512, 100]; // divergent positions
    let b = seeds.len();

    // SOLO: each slot at its OWN single position (region 0, fresh KV).
    let mut solo: Vec<u32> = Vec::with_capacity(b);
    for bi in 0..b {
        engine.multiseq_arena = None;
        let logits = engine
            .forward_tokens_multiseq_logits(&[seeds[bi]], &[start_pos[bi]], &[0], max_seq)
            .expect("solo divergent logits");
        assert_eq!(
            logits[0].len(),
            engine.config.vocab_size,
            "solo logits must be full-vocab"
        );
        solo.push(argmax(&logits[0]));
    }

    // BATCHED: all B at their divergent positions, stable regions [0..b).
    engine.multiseq_arena = None;
    let regions: Vec<usize> = (0..b).collect();
    let logits = engine
        .forward_tokens_multiseq_logits(&seeds, &start_pos, &regions, max_seq)
        .expect("batched divergent logits");
    for l in &logits {
        assert_eq!(
            l.len(),
            engine.config.vocab_size,
            "batched logits must be full-vocab"
        );
    }
    let batched: Vec<u32> = logits.iter().map(|l| argmax(l)).collect();

    for bi in 0..b {
        assert_eq!(
            solo[bi], batched[bi],
            "slot {bi} @pos {}: batched Q4_K LM-head argmax {} != solo {}",
            start_pos[bi], batched[bi], solo[bi]
        );
    }
    println!("[multiseq-q4k-lmhead] divergent-pos B={b}: batched=={solo:?}");
}
