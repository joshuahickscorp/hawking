#![cfg(target_os = "macos")]
//! C1 parity: SINGLE-TCB tail for the multiseq decode path.
//!
//! C1 makes `forward_tokens_multiseq_stack_tcb` stop committing and RETURN the
//! live TokenCommandBuffer; under DISMANTLE_QWEN_Q4K_LMHEAD=1 the LM-head Q4_K
//! GEMM is appended into THAT same command buffer and a SINGLE commit_and_wait
//! covers layers + LM head (was two separate commits/round-trips). This is pure
//! plumbing — same dispatches, same order, one fewer GPU submit+fence — so the
//! token/logit output must be UNCHANGED. This test pins that:
//!
//!   (A) ANCHOR — B=1 multiseq's first token (single-fused-TCB R1 path) ==
//!       the proven single-stream `forward_token_greedy_tcb` first token, under
//!       the SAME full-vocab Q4_K head. Proves the fused tail did not perturb the
//!       layers -> LM head -> argmax result. (argmax-equality: single-stream runs
//!       the LM head as a v3_8r GEMV while multiseq runs it as the v3w batched
//!       GEMM at B=1 over the SAME weight — argmax-identical by the R1 landing,
//!       so we compare token ids, not raw logits, on this leg.)
//!
//!   (B) NO-CONTAMINATION + BIT-IDENTICAL LOGITS — B distinct slots decoded
//!       TOGETHER (one fused commit per step) give each column logits that are
//!       BIT-IDENTICAL (full f32 equality, not just argmax) to that slot decoded
//!       ALONE at B=1 through the identical fused path, over several steps. Both
//!       sides use the SAME v3w batched GEMM for projections AND LM head, so a
//!       correct single-TCB fusion is bit-exact; a stale-read / wrong-fence bug
//!       introduced by folding the LM head into the layer command buffer (reading
//!       x_norm before the trailing add+rmsnorm completed) would diverge here.
//!
//!   (C) DETERMINISM — re-running the SAME batched step yields BIT-IDENTICAL
//!       logits. A missing fence after the fused commit would surface as run-to-
//!       run jitter; this catches it directly.
//!
//! Sets DISMANTLE_QWEN_Q4K_LMHEAD=1 BEFORE load (the full-vocab Q4_K head is
//! built at load time), so it must run in its OWN test binary, single-threaded.
//! #[ignore] because it loads the 1.93 GB Qwen-3B model. Clean-skips if weights
//! are missing.
//!
//! Run:
//!   cargo test --release -p dismantle-core --test multiseq_single_tcb_tail_parity \
//!     -- --ignored --test-threads=1 --nocapture

use std::path::PathBuf;

use dismantle_core::{model::qwen_dense::QwenDense, profile::fresh_test_profile, Engine, EngineConfig};

fn weights_path() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

/// Load Qwen-3B with the GPU Q4_K LM head FORCED ON (the C1 single-TCB-tail path
/// only fuses the LM head when this flag is set). Every other lever pinned to its
/// default so the only thing under test is the single-commit fused tail.
fn load_q4k_lmhead() -> Option<QwenDense> {
    let w = weights_path();
    if !w.exists() {
        eprintln!("skipping multiseq_single_tcb_tail_parity: weights missing at {w:?}");
        return None;
    }
    for v in [
        "DISMANTLE_QWEN_VOCAB_PRUNE",
        "DISMANTLE_QWEN_VOCAB_PRUNE_CORPUS",
        "DISMANTLE_QWEN_F16_KV",
        "DISMANTLE_QWEN_FLASH_ATTN",
        "DISMANTLE_QWEN_W4A8",
    ] {
        std::env::remove_var(v);
    }
    std::env::set_var("DISMANTLE_QWEN_Q4K_LMHEAD", "1"); // env_on => exact "1"; full-vocab Q4_K head
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

    // (A) ANCHOR: B=1 fused-tail multiseq first token == single-stream first
    // token, under the SAME full-vocab Q4_K head.
    for &s in &seeds {
        engine.kv.reset();
        let single = engine.forward_token_greedy_tcb(s, 0).expect("single-stream fwd");
        engine.multiseq_arena = None;
        let ms = engine
            .forward_tokens_multiseq(&[s], &[0], max_seq)
            .expect("B=1 multiseq fused-tail")[0];
        assert_eq!(
            single, ms,
            "anchor: B=1 fused-tail multiseq token {ms} != single-stream {single} (seed {s})"
        );
    }

    // (B) NO-CONTAMINATION + BIT-IDENTICAL LOGITS. SOLO references: each seed
    // decoded ALONE at B=1 through the fused-tail R1 path, capturing FULL logits.
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

    // BATCHED: all B seeds decoded TOGETHER (one fused commit per step), lockstep.
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
    // sides => bit-exact; any stale x_norm read from the fused tail diverges).
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

    // (C) DETERMINISM: re-run the FIRST batched step (fresh KV) and demand
    // bit-identical logits to the first batched step above.
    engine.multiseq_arena = None;
    let positions0 = vec![0usize; b];
    let rerun = engine
        .forward_tokens_multiseq_logits(&seeds, &positions0, &regions, max_seq)
        .expect("rerun fused-tail logits");
    for bi in 0..b {
        let first = &batched_logits[bi][0];
        assert_eq!(rerun[bi].len(), first.len(), "slot {bi}: rerun length differs");
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
    println!(
        "[multiseq-single-tcb-tail] B={b} anchored + batched logits BIT-IDENTICAL to solo over {n} steps + deterministic: {batched_tokens:?}"
    );
}
