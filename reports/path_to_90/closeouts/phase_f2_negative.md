# Phase F.2 — NEGATIVE (acceptance miss confirmed, plateau-then-overfit)

**Halted at:** 2026-05-20 12:28 (early kill, e3 shard 2 of 60)
**Halted on:** acceptance miss decision rule (e2 head 7 top1 < 5.5%)
**Branch:** `claude/dreamy-golick-d54ff8`
**Canonical ckpt:** `eagle4/checkpoints/medusa_v1/best.npz` (epoch 1, top1_mean=0.172)

## Root cause

The medusa K=8 head architecture + slope=2.0 weighting converged at
epoch 1 with `top1_mean=0.172`. Epoch 2 regressed (`0.166`) — same
overfit pattern observed on the 100k subset's e1→e2. Five-times the
training data did NOT prevent convergence at a sub-threshold local
minimum. Acceptance bar (head[K-1] top1 ≥ 15%) reached only **5.0%**
on the best epoch — three-fold short.

The structural finding: **late medusa heads in this architecture do
not learn beyond a ~5% top-1 ceiling on this substrate.** Adding
data, weighting, and KL distillation did not move the floor. The
limitation is information-theoretic — predicting token N+7 from a
single hidden vector at position N is much harder than predicting
N+1, and the present adapter (RMSNorm → SwiGLU → tied LM head)
cannot extract the missing information from `hidden_high` alone.

## Three-epoch trajectory (canonical evidence)

Per `eagle4/checkpoints/medusa_v1/best_eval.json` + log greps:

| head | e0 top1 | e1 top1 | e2 top1 | e0 top10 | e1 top10 | e2 top10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 60.5% | **59.9%** | 59.5% | 87.6% | **87.6%** | — |
| 1 | 28.0% | **27.3%** | 26.3% | 56.9% | **56.6%** | — |
| 2 | 14.7% | **15.1%** | 14.4% | 40.5% | **39.9%** | — |
| 3 | 10.2% | **10.4%** | 10.0% | 34.6% | **33.3%** | — |
| 4 | 7.0% | **8.0%** | 7.6% | 30.1% | **28.6%** | — |
| 5 | 6.0% | **6.1%** | 5.7% | 27.3% | **25.9%** | — |
| 6 | 5.9% | **5.8%** | 4.9% | 26.6% | **26.2%** | — |
| 7 | 5.3% | **5.0%** | 4.3% | 26.1% | **25.0%** | — |
| **mean** | **17.2%** | **17.2%** | **16.6%** | — | — | — |

**bold** = best-epoch numbers per head (almost uniformly e0 or e1).
e1 had the best `top1_mean`, so e1's ckpt is `best.npz`.

## Acceptance gate

| Criterion | Bar | e1 (best) | Verdict |
| --- | --- | --- | --- |
| top1[0] ≥ 40% | 40% | 59.9% | ✅ massive headroom |
| top1[K-1] ≥ 15% | 15% | **5.0%** | ❌ 3.0× short |

Head 0 ships acceptance comfortably; head K-1 misses by 10 percentage
points. The audit-prescribed bar is NOT met. F.3 (Rust port) is NOT
authorized against this checkpoint per the F.5 plan's acceptance
contract.

## What ran

- Subset (default hyperparams, 100k × 3 epochs): completed in 46 min
  on 2026-05-19. best e1: top1_mean=0.156, head 0=60.1%, head 7=4.1%.
- Subset retune (slope=2.0, 100k × 3 epochs): launched, killed mid-run
  by user before completion.
- Full run #1 (slope=2.0, 491k × 10 epochs, background): launched and
  killed at e0 s1850 (~30 min, no ckpt). RAM contention.
- Full run #2 (slope=2.0, 491k × 10 epochs, background): launched and
  killed at e0 s1850 (~30 min, no ckpt). Background-task kill pattern.
- Full run #3 (slope=2.0, 491k × 5 epochs, **foreground** with ETA
  injection): completed e0+e1+e2, killed at e3 s120 (~2h47m total).
  e1 was the convergence point; e2 confirmed regression. PID 96226.

## What unblocks F.2 (future work, NOT this session)

Three avenues from cheapest to most ambitious:

1. **F.2 v2 — richer hidden input.** Add `hidden_low_+i` and
   `hidden_mid_+i` columns via another window-join over `eagle4_v0`
   shards (cheap, ~30 sec per Pattern 11). The current adapter sees
   only `hidden_high` at position N; multi-layer hidden may carry the
   information late heads need. Quality-of-hypothesis: medium.

2. **F.2 v3 — multi-token context window.** Adapter input becomes
   `concat(hidden_high_N-k..N)` for some k. Gives the late heads
   N-step context instead of single-vector. Cost: re-architecture
   plus new training run.

3. **Pivot to Eagle4 + medusa hybrid.** Use medusa heads 0..2 (top1
   28%, 15%, 10%) as cheap parallel drafts for K=3 lookahead, fall
   back to Eagle4 chain for deeper positions. Heads 3..7 don't ship.
   Quality-of-hypothesis: high — early heads ARE above ship bar
   (head 2 at 15% top1, head 1 at 27%). This pivots F.2 from "K=8
   wide" to "K=3 narrow" which the F.1 capture already supports
   (just read fewer columns from the same shards).

## Path-to-100 trajectory update

Per `path_to_100_retool.md`, F.2 was scheduled to contribute +30
realistic / +40 best dec_tps. **With this acceptance miss, F.2
ships zero** unless a narrow-K hybrid (path #3 above) is pursued.

Revised math:

| Lever | Best | Realistic | Status |
| --- | ---: | ---: | --- |
| Baseline | 26.78 | 26.78 | shipped |
| L7 (kernel rewrites) | +30 | +20 | plan ready |
| L5 (chain pipeline) | +8 | +5 | plan ready |
| F.2 narrow-K hybrid (3 heads, future) | +15 | +10 | speculative |
| **Subtotal w/ stretch** | **+53** | **+35** | |
| **Projected dec_tps** | **80** | **62** | |

100 dec_tps is no longer in reach without F.2 v2/v3 architectural
work. Realistic ceiling drops to **~62 dec_tps**. Best-case stretch
hits **~80 dec_tps** if narrow-K hybrid lands.

This still beats llama.cpp's ~20 dec_tps headline on M3 Pro by 3×,
which keeps the project's defensible ship narrative intact even if
the 100-tps banner falls.

## Followups for next session

- Read this closeout + `methodology_distilled_post_f2.md` (Patterns
  11–20) + `l7_tomorrow_pickup.md`. The L7 pickup doc is the
  next-session entry point.
- L7 is now the primary lever. Per the pickup doc, start with
  `moe_expert_pair_fused.metal` (highest leverage).
- F.2 v2 (hidden_low/mid columns) is a defensible alternative path
  if user wants to revisit medusa. The substrate (`eagle4_v0/`) is
  preserved on disk; the window-join script (`rewrite_medusa.py`)
  needs <100 lines of edit to project additional columns.
- `eagle4/checkpoints/medusa_v1/epoch_00.npz`, `epoch_01.npz`,
  `epoch_02.npz` can be cleaned (each 805 MB = 2.4 GB freed). The
  `best.npz` (= `epoch_01.npz` bytes) preserves the canonical
  checkpoint.

## Methodology lessons added (Pattern 21)

**Pattern 21 — Plateau-then-overfit signals architecture limit, not
data limit.** Subset showed e1→e2 regression on 100k rows. The 5×
data full run showed the SAME e1→e2 regression at full scale. When
the regression pattern survives data scaling, the issue is
information-theoretic in the architecture, not statistical in the
sample size. Append to `methodology_distilled_post_f2.md` for next
session.

## Audit trail

- F.2 infra commit: `a25ce6a` (2026-05-19)
- F.2 deferred closeout (intermediate, superseded by this doc):
  `phase_f2_deferred.md` — kept on disk for chain-of-evidence
- Best ckpt: `eagle4/checkpoints/medusa_v1/best.npz` SHA: read from
  disk if needed
- best_eval.json: epoch 1, top1_mean=0.17207935455828352
- Full kill timeline:
  - 2026-05-19 ~22:00: subset complete (canonical baseline)
  - 2026-05-19 ~22:30: retune killed mid-run by user
  - 2026-05-19 ~23:30: full #1 killed at e0 s1850 (RAM contention)
  - 2026-05-20 00:05: full #2 killed at e0 s1850 (background pattern)
  - 2026-05-20 09:32: full #3 launched (foreground, 5 epochs)
  - 2026-05-20 10:32: e0 eval landed, best.npz saved
  - 2026-05-20 11:31: e1 eval landed, best.npz updated
  - 2026-05-20 12:27: e2 eval landed, regression confirmed
  - 2026-05-20 12:28: kill issued, training stopped at e3 s120
