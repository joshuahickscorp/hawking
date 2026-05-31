# Session closeout — 2026-05-31 (evening): L3.1 moat sweep + parallel lanes + clean-room baselines

**Branch:** `codex/maximal-spec-colab` — **26 commits ahead of origin, all LOCAL (unpushed).**
**Final state:** tree clean except 3 untracked junk files (`on_smoke.err`, `server_smoke.err`, `traces/`).
**Opened against:** `plans/handoff_next_session_2026_05_31.md` (the morning handoff).

## The headline — three MEASURED numbers (clean, user-run, uncontaminated)
`tools/bench/clean_room_batch.sh`, Claude quit. These close two questions that were distorting every
projection (`reports/clean_room_baselines_2026_05_31.md`):
- **Decode anchor RESOLVED → ~31** (clean 29.12 tps, 256 tok greedy). ~39 was optimistic (−25%).
  Closes bible §3.0 Correction 2. *The §3 envelope is now SUPERSEDED but NOT re-projected — that's the
  one attended-strategy item left.*
- **Q3_K byte-cut DEAD, re-confirmed** (33.3→34.7 GB/s = 22–23% peak; Q3 slower than Q4 on the FFN
  shapes that dominate decode). **QTIP is the sole byte-cut path.**
- **Energy floor = 0.17 J/token** (3.73 W GPU / 4.87 W package) — the §8 L4.2 axis now has a baseline.

## What landed (this session's commits, oldest→newest)
- `a0f7a6e` — **Oracle 1 (draft tuning) GO**: per-session warm-start τ_warm_suffix 3.40 pooled / 2.33
  median (+0.89 additive over the shipped prefix cache, prefix-cache-discounted). ESTIMATE (git proxy).
- `cac51ba` — **QTIP byte-cut design** + the **clean-room runbook** (`clean_room_batch.sh`).
- `931613f` — **Oracle 2 (semantic) NO-GO Type-2** (+1.48 mean / +0.00 median; parked behind a
  real-session-logs re-run) + **Oracle 3 (vocab screen) NO-GO Type-1** (0% certified; norm/frequency
  anti-correlation) + findings.
- `680cb35` — **merge: L3.1 draft-tuning body** (`usage_capture` observer + `UserNgramDraft`, default-off).
  Parity **bit-identical, re-verified by me in main** (draft_accepted=7); 91/91 lib tests green.
- `3b4f6f2` — **P1 prefill-MMA: GO but deferred** (handoff + 3 Kill-Protocol entries).
- `618024b` — **fix:** stale profile shader-hash + `measure_joules.sh` f-string (unblocked clean-room B/C).
- `d6733f8` — **Q3 byte-cut clean-confirmed NO-GO** (dead_levers + design banner).
- `010bc4a` — **clean baselines + bible §3.0 anchor closure.**

## Decisions / kills recorded (`reports/dead_levers.md`)
- **Semantic cache (L1.2 ext)** — Type-2, PARKED (re-gate on real file-interleaved session logs; oracle built).
- **Vocab screen (L3.1)** — Type-1 DEAD (full-rank head + norm/frequency anti-correlation; cert needs cos>1).
- **Q3_K sub-Q4 byte-cut** — Type-1 DEAD (compute-bound, not BW-bound). QTIP is the reframe.
- **Q4_K batched MMA on rows≤cols** — Type-1 occupancy (tall-shape MMA is GO but dormant in the predec path).

## In flight / preserved (NOT lost)
- **Draft tuning** — merged, lossless, **default-OFF**. Turning it on is **−78%** (paired) because
  `forward_tokens_verify` falls back to a CPU full-vocab pass at ~18% acceptance. **Named unblock: the
  pruned-Q4K batched-verify fast-path.** The `usage_capture` observer is free and landed.
- **P1 prefill-MMA** — GO (+22–24% on the tall ffn gate/up GEMM, parity-green *agent-reported*), **NOT
  merged**: stale worktree base (degenerate `qwen_dense.rs` conflict) + the MMA is v3w-layout while the
  shipped batched path is predec → dormant where it wins. Branch `worktree-agent-a08c1cb44eb3d4e47`
  (`c9b1c07`) + `stash@{0}` preserved; recipe in `plans/p1_prefill_mma_integration_handoff_2026_05_31.md`.

## The re-baselined forward picture (honest)
Decode is **~31 at the kernel ceiling**. The two source lanes both landed **validated-but-blocked**, not
as shipped tps. Forward throughput rests on **two named moves**:
1. **QTIP** (byte-cut) — behind its offline quality oracle (recon-RMSE vs Q4_K at 3 bits, QTIP-*from-f16*).
   No kernel until it clears.
2. **Pruned-Q4K batched-verify fast-path** (spec) — unblocks the −78% draft-tuning penalty. Highest-leverage
   decode-tps unblock.

## Owed / open (for the next session)
- **Thermal-median** (`clean_bench.sh ×N`) to pin the canonical anchor at 29 vs 31 (tightening, not reopening).
- **§3 envelope re-projection from ~31** + the genuine strategic call: **does ~50 dense stay the target?**
  At a measured ~31 ceiling with kernels closed, ~50 needs QTIP's full multiplier **and** the spec unblock
  both landing — reachable, but no longer "high-confidence parity."
- **§6 / roadmap consistency pass** — the ~7 "path-to-50" lines are now *doubly* stale (kernels closed +
  anchor moved); the rewrite is unblocked the moment the goalpost question is decided.

## The arc worth naming
Across these sessions the **extract** levers kept dying (Q3, KV-working-set, the four weight-structure
kills, semantic cache, vocab screen, decode-kernel micro-opts) and the **build** levers kept surviving
(prefix cache shipped, draft tuning merged-lossless, stateful stubs). 8+ kills for the cost of NumPy
afternoons, 2 real shipped/merged wins, **zero contaminated results** (every absolute number was paired
or user-run). The program has sorted itself along that line.

**Push decision:** 26 commits local-only; user pushes when ready.
**Next-session opener:** `plans/next_session_opener_2026_05_31_evening.md`.
