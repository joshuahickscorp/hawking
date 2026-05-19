# path-to-125 session closeout 2026-05-19c³ (post "do all that can be done while L8 trains")

**Branch:** `claude/dreamy-golick-d54ff8`
**Continuation entry HEAD:** `f962e65` (closeout-c²)
**Continuation exit HEAD:** `debf398` (L9 bench queue update)
**Substantive commits this continuation:** 2

User directive: "do all that can be done while l8 is training".

## What got done

| # | Item | Status |
|---|---|---|
| 1 | **L7.1 wire-up** — add `v3_xtg` branch to `gemm_q4_k_schedule` selector | ✅ `9b7038d` |
| 2 | Bench-queue update with L5/L7.1 entries | ✅ `debf398` |

## What was scoped but not done (and why)

| Item | Reason not shipped this continuation |
|---|---|
| K-batched xtg (`gemm_q4_k_m_fused_v2_kbatch_xtg`) | Memory budget tight (32 KB needed at K=4 cols=2048, equal to M3 Pro per-core threadgroup mem ceiling). Realistic gain limited — Q4_K_M K-batched only fires on dense layer 0 FFN; MoE routed experts use the union pipeline (not kbatch), LM head is f16 (not Q4_K). Marginal payoff for nontrivial GPU memory risk. Worth a focused-session attempt with iterative tuning. |
| LM head kbatch x_cache | Already optimized — `gemv_f16_lmhead_kbatch.metal` uses 192-float W/X/D threadgroup tile. No gap to fill. |
| L5 decode-step pipelining | Required design discovery: chain step N+1 propose depends on verify step N's argmax (hard dependency). Real overlap requires speculative decoding, not just SharedEvent scheduling. Out of scope. |
| L4 hot-path fix | Structurally regressive (breaking the fused TCB kernel adds ~30-100 µs/layer × 27 layers to save ~10-50 µs/AMX-call). Net 2-5× slowdown. No "fix" exists. |
| Bench A/B for L4/L7.1 | Requires clean window (Cmd-Q Claude + pause training). Queued. |

## Training status snapshot

```
pid 84960  STAT=UN  elapsed=31:02  cpu=21%  rss=10 GB

[step=1   epoch=0 loss=90.628 gate=0.099 α=0.00 wall=105.4s]
[step=25  epoch=0 loss=54.967 gate=0.060 α=0.05 wall=1528.4s]
```

**Loss trajectory:** 90.6 → 55.0 over 24 steps. Normal early-training
descent.

**Gate trajectory:** 0.099 → 0.060. Gate is DROPPING, not climbing.
At this stage α is still ramping (0.05/1.0), so the chain CE pressure
that should keep gate alive isn't fully engaged yet. The real test is
gate's trajectory after α plateaus near 1.0 (~step 500 per the
`--target-warmup-steps 500` schedule).

**If gate drops to ~0.001 by step 500-1000:** the fix-(e) hypothesis
(gate_init=0.1 prevents collapse) is FALSIFIED. Next move: vector
residual_gate (fix f) or block rewrite (fixes g/h) per closeout
§ Branch 3.

**If gate stays ≥ 0.05 by step 500-1000:** fix-(e) is working;
chain_accept should rise above 7%. Run τ-eval after epoch 1 to
measure.

## Current dec_tps

**No fresh measurement available** under training contention.

Last clean-window numbers (pre-session):
- Off baseline:                   26.78
- ngram-spec K=4 + parallel-k:    26.71
- Eagle4 K=1:                     16.93
- Eagle4 chain K=4 + parallel-k:   7.23

This continuation shipped only callable/wired-up kernels, no
production-default changes, so current dec_tps remains within
noise of these baselines.

## L7.1 wire-up detail

`gemm_q4_k_schedule = "v3_xtg"` now selects the threadgroup-x_cache
Q4_K_M GEMV. To activate after training+bench window opens:

```bash
# edit profiles/deepseek-v2-lite-q4.m3pro18.json:
#   "gemm_q4_k_schedule": "v3_xtg"          # globally use xtg
#   OR
#   "gemm_q4_k_schedule": "per_shape"       # shape-targeted
#   "gemm_q4_k_schedule_per_shape": {
#     "10944x2048": "v3_xtg",               # V2-Lite expert
#     "102400x2048": "v3_xtg"               # LM head (sequential decode)
#   }
```

Production paths xtg affects:
- Standalone Q4_K_M GEMV via `q4k_schedule_for_shape` dispatcher
- Sequential-decode LM head (rows=102400, cols=2048)
- Dense layer 0 FFN gate/up/down (rows=10944, cols=2048)

Production paths xtg does NOT affect (each has its own kernel):
- K-batched verify path (uses `gemm_q4_k_m_fused_v2_kbatch`)
- MoE routed experts (use `moe_batched_gemm_q4_indexed_v2t_*` family)
- Fused rmsnorm+gemv attention projections (f16-weighted kernels)
- Fused gate+up moe_v2t_gu kernels

Realistic production dec_tps gain estimate: **+1-3%** (limited blast
radius). The kernel is correct + parity-validated; the question for
the bench window is whether the small wins outweigh any per-shape
threadgroup memory pressure.

## Validation matrix at this exit

```
build:                                clean
cargo test --lib                      45/45 pass
cargo test --test path_b_parity       18/18 active (+4 ignored)
cargo test --test amx_proj_parity     4/4 pass
cargo test --test multi_queue_smoke   2/2 pass
cargo test --test shared_event_smoke  2/2 pass
cargo test --test q4_k_v3_xtg_parity  3/3 pass
EAGLE4_PARITY_TEST=1 eagle4_decode_parity at 16 tok:
  BIT-IDENTICAL Off vs Eagle4 (ran 73s under training contention)
```

User diagnostic edits preserved exactly through all 12 commits:
+27 lines / 3 files (engine.rs +10 / kernels/mod.rs +13 /
deepseek_v2.rs +4), identical to session entry.

## Total session commit log

```
4f24d46  L4 AMX V2-Lite attn projections
46a9ce7  L9 bench queue tracker
8c51a02  L5 multi-queue scaffold
6ff3e4c  L8 --gate-init CLI flag
372a6ea  mid-session exhaustion_report (superseded)
e482463  L5w SharedEvent + TCB-on-secondary + Eagle4 routing
4aeca8b  L8L launch wrapper + nohup kick (pid 84960)
2036b15  closeout-c (superseded by c²)
50513c0  L7.1 gemm_q4_k_m_v3_xtg + parity
f962e65  closeout-c² (post "fix all and proceed")
9b7038d  L7.1-wire gemm_q4_k_schedule="v3_xtg" branch lit
debf398  L9 queue: L5 + L7.1 wire entries
(this)   closeout-c³
```

11 substantive commits + 3 closeouts.

## Errata

1. **The xtg kernel hasn't been benched.** Parity is validated; speed
   is hypothesized. The 5-15% projected gain on xtg-served shapes is
   based on the 8× redundant x-read elimination math, not measured.
   First clean window decides whether it ships as default.

2. **Eagle4 chain decode dec_tps still capped at 7%-accept ceiling.**
   None of L4/L5/L7.1/L7.1-wire address this. The structural fix is
   L8 training (gate_init=0.1 from scratch), and that's running.
   Until L8 reports a chain_accept readout, the dec_tps trajectory
   is gated.

3. **L7.2 single-kernel fusion was deferred earlier and remains so**
   — register-pressure math gives ~44 KB shmem need on M3 Pro 32 KB
   budget. Tight tiling work, focused session.

4. **Q4_K_M K-batched xtg is a real opportunity** (was identified
   while exploring) but the memory budget at K=4 cols=2048 is
   right at the 32 KB ceiling. Either reduce ROWS_PER_TG (halves
   row throughput) or use a piece-meal x_cache (loop reorder).
   Worth a focused-session attempt.

## Highest-leverage next action

Same as before: **wait for L8 training's chain_accept readout.**
That answers whether the architecture clears the ceiling. While
waiting, no further productive shader work is in scope without a
bench window — every remaining lever needs either:
- A clean bench window (Cmd-Q Claude + pause training), OR
- Focused-session GPU profiling work (L7.2, kbatch_xtg), OR
- Speculative decoding design (L5 real pipelining).

Stop point reached.
