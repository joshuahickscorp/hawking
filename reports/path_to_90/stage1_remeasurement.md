# Path-to-90 step 10 re-measurement

**Captured:** 2026-05-18 20:10–20:12 EDT
**Commit benched:** `93025ba` (post-`acca22d` GPU h_shared)
**Host:** M3 Pro 18 GB
**Workload:** 3 prompts × 16-token greedy decode, `--speculate off` vs
`--speculate eagle4`, via `tools/bench/stage1_remeasurement.sh` (clean
window: Claude.app quit, slm idle). Decode-only `dec_tps` parsed from
dismantle's `[stats]` line (the methodology fix called out in the prior
report).

## The numbers

| prompt | Off dec_tps | Eagle4 dec_tps | Eagle4 accept |
|---|---:|---:|---:|
| "The quick brown fox" | 27.02 | 8.17 | 15/16 |
| "Once upon a time" | 27.15 | 7.36 | 14/16 |
| "def fibonacci(n):" | 27.11 | 7.78 | 14/16 |
| **median** | **27.10** | **7.36** | — |
| **total accept** | — | — | **43/48 = 89.6 %** |

Off baseline is rock-stable across prompts (27.02 / 27.15 / 27.11 — sub-
percent variance). Matches the Stage 0 baseline 26.93 from commit
`72e3926`. Confirms the Wedge C path is unchanged.

Eagle4 acceptance lands at 89.6 % — slightly above eagle4's trained
87.48 % held-out number. Three reasons that's plausible:

- The three prompts here are short and template-y ("The quick brown
  fox…" is essentially memorized); steady-state on Spec-Bench would
  land lower.
- 48 drafts is a small sample; ±5 % stderr at this scale.
- GPU-sourced hiddens (this session's three follow-up commits) put
  the head firmly back in-distribution with its MLX bf16 training
  data — that's the structural reason for the ~85× jump from the
  pre-fix 2.1 %.

## Full architectural progression

```
Stage 1 dec_tps for Eagle4 mode (M3 Pro 18 GB, 16-token greedy):

  step 8  (CPU walk)           : 0.54  (2 % accept)
  step 10-fa (GPU capture)     : 1.89  (94 % accept)
  step 10-fb (GPU lm_head arg) : 6.52  (94 % accept)
  step 10-fc (GPU h_shared)    : 7.36  (90 % accept)  ← THIS RUN
  ───────────────────────────────────────────────────────────
  step 7    (full Metal head)  : ~15-18  (projected)
  step 1+Stage0.5 (MLX)        : ~25-30  (projected, gates Stage 5)
  Stage 1 block-ship band      : 18-24
```

Headline: **13.6× faster than the original CPU-walk decode**, in 4
follow-up commits over ~3 hours of architectural work.

## Block-ship gate

Stage 1 lower bound: 18 dec_tps. Observed: 7.36. **Gate: HALT** but
well-bounded — the remaining cost is fully attributable to one
specific code path (Eagle4Head's CPU forward for the in_proj /
block_attn / block_mlp / mask / calib gemvs, ~80 ms/token).

Per-token cost breakdown (decode_ms ~136 / 16 tokens = ~136 ms):

| component | ms/token | status |
|---|---:|---|
| Wedge C V2-Lite forward (per-layer commits) | ~41 | shipped (`679c077`) |
| Eagle4Head CPU forward (in_proj + block + mask + calib) | ~80 | **step 7 target** |
| GPU lm_head argmax (`gemv_f16_argmax_dispatch`) | ~10 | shipped (`808d8db`) |
| h_shared (read `moe_shared_out_buf`) | ~0 | shipped (`acca22d`) |
| measurement noise / overlap | ~5 | — |

Step 7 (Metal-accelerated Eagle4Head forward) is the next architectural
unlock. Expected outcome: Eagle4 dec_tps lands in the 15-18 range,
clearing the Stage 1 lower bound on coherent code prompts and
landing the Stage 1 block-ship.

## Decision

1. **Proceed to step 7.** Pin the eagle4 head weights as Metal buffers
   at engine load; route all 10 head GEMVs through dismantle's
   existing Metal helpers (`gemv_f32_metal`, `gemv_f32_attn_dispatch`,
   etc.). RMSNorms move GPU-side. SiLU stays CPU for the small
   intermediate vectors (or fuse into the gemv kernels).
2. After step 7 lands: re-run `stage1_remeasurement.sh`. If ≥ 18
   tok/s, Stage 1 ships and we proceed to step 11 (routing recall
   fine-tune, parallelizable Python work) + steps 12-17 (Path B
   kernels for Stage 2's 38-50 tok/s target).
3. If step 7 doesn't clear 18 tok/s, the Stage 0.5 MLX-pattern
   adoption mandate from step 2 (commit `48be7a1`) jumps the
   queue. Bumps Off baseline from 27 to ~55-65 tok/s; cascades to
   Eagle4 ~25 tok/s by elementary arithmetic.

## Artifacts

```
reports/path_to_90/_stage1_remeasurement/
  STATUS.log               ← full script log
  raw.json                 ← parsed metrics (dec_tps + accept)
  off_t0.log .. off_t2.log ← Off-mode output incl. [stats]
  e4_t0.log  .. e4_t2.log  ← Eagle4-mode output incl. per-step accept/reject
```
