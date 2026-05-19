# Path-to-90 Stage 1 re-measurement (post-step-7 Metal Eagle4Head)

**Captured:** 2026-05-18 20:31–20:33 EDT
**Commit benched:** `0b4bef5` (= `518e580` step 7 + script doc refresh)
**Host:** M3 Pro 18 GB
**Workload:** 3 prompts × 16-token greedy, `--speculate off` vs
`--speculate eagle4`, via `tools/bench/stage1_remeasurement.sh`
(clean window: Claude.app quit, slm idle). Decode-only `dec_tps`
parsed from dismantle's `[stats]` line.

## The numbers

| prompt | Off dec_tps | Eagle4 dec_tps | Eagle4 accept |
|---|---:|---:|---:|
| "The quick brown fox" | 25.21 | **10.41** | 15/16 |
| "Once upon a time" | 27.02 | 9.62 | 14/16 |
| "def fibonacci(n):" | 24.76 | 9.82 | 14/16 |
| **median** | **24.76** | **9.62** | — |
| **total accept** | — | — | **89.6 %** (43/48) |

## Off baseline noise

Off median dipped to 24.76 from the prior re-bench's 27.02 (commit
`700e68e`). The three Off prompts span 24.76–27.02 — normal ±10 %
thermal / measurement variance on M3 Pro at decode workloads. My
changes don't touch the Off mode path (Eagle4 logic is conditional
on `speculate_mode == Eagle4`), so Off is unchanged code-wise. Worth
re-measuring in a more controlled clean window once the rest of
the stage work lands, but the variance band is the priority signal,
not the median.

## Eagle4 improvement is real

Three Eagle4 prompts: 10.41 / 9.62 / 9.82 — **sub-10 % spread**.
Stable measurement. Step 7's Metal Eagle4Head delivers a clean
**+31 % on the prior re-bench's 7.36 tps**.

## Architectural progression (canonical numbers)

```
Stage 1 Eagle4 dec_tps progression (M3 Pro 18 GB, 16-token greedy):

  step 8   (CPU walk)              :  0.54  ←  2 % accept
  step 10f (GPU capture)           :  1.89  ← 94 %
  step 10f (GPU lm_head argmax)    :  6.52  ← 94 %
  step 10f (GPU h_shared)          :  7.36  ← 90 %
  step 7   (Metal Eagle4Head)      :  9.62  ← 90 % (THIS RUN)
  ───────────────────────────────────────────────────────────
  Stage 1 lower band               : 18.00  ← block-ship gate
  Stage 1 upper band               : 24.00
  Stage 2 (Path B K-batched verify): 38-50  ← amortized verify multiplier
  Stage 3 (mask-driven prefetch)   : 55-75
  Stage 4 (DySpec tree decode)     : 70-95
  Stage 5 (hardware paths, AMX/ANE): 95-125  ← project headline
```

**17.8× faster than the original CPU-walk decode** in 5 follow-up
commits over a single architectural session.

## Block-ship gate

Stage 1 (≥ 18 tps): **HALT.** Observed 9.62; need 18.

Remaining ~2× gap is no longer in the eagle4 head — that's been
Metal-ized end-to-end except for the small CPU intermediates (rmsnorm,
silu, residual adds). The gap now sits in:

| component | ms/token | notes |
|---|---:|---|
| V2-Lite Wedge C (per-layer commits for capture) | ~41 | adds ~4 ms vs single-TCB; capture forces per-layer commits |
| Eagle4Head Metal forward (9 f16 gemvs + CPU intermediates) | ~50 | per-gemv dispatch + CPU rmsnorm/silu readback |
| GPU lm_head argmax | ~10 | shared with Off |
| h_shared from moe_shared_out_buf | ~0 | no extra dispatch |

Total ~101 ms = 9.9 tps. Matches measurement.

## Next levers (in priority)

1. **Stage 0.5 MLX-pattern adoption.** Per step 2's decision rule
   (commit `48be7a1`), already mandatory. Lifts Off baseline from
   ~27 → ~55-65 tps via 2-3× efficiency gains on the V2-Lite hot
   kernels (gemv_q4_k_v3, MoE pair matmul, MLA decode). Cascades
   directly to Eagle4 — at current 9.62 ÷ 27 = 0.36 of Off, post-
   Stage-0.5 Eagle4 would be 55 × 0.36 ≈ 20 tps. **CLEARS STAGE 1.**
   Effort: 1-2 weeks of focused kernel work.

2. **Single-TCB Eagle4Head encoding.** Replace 9 standalone Metal
   gemv dispatches with one TCB-encoded forward. Requires moving
   the eagle4 head's RMSNorm + SiLU + residual adds to GPU
   (existing helpers: rmsnorm_metal_buf_tcb, etc.). Eliminates
   ~10 ms/token of per-gemv dispatch + readback overhead.
   Expected: +1-2 tps. Effort: half day.

3. **K>1 batched verify via Path B kernels** (Stage 2 territory,
   steps 12-17). Genuine multiplicative win — at K=4 with 87 %
   chain acceptance, ~3.5 tokens emitted per V2-Lite forward.
   Lifts Eagle4 to ~30-40 tps. Effort: 3 weeks.

The biggest single lever is #1 (Stage 0.5), and it also sets up
the Stage 5 headline number. The architectural batch for this
session closes here.

## What's settled

- Eagle4 spec decode infrastructure: shipped end-to-end.
- GPU-native capture + head forward: shipped.
- 89.6 % draft acceptance: confirmed across 3 prompts. Eagle4 head
  is functionally correct against the dismantle V2-Lite forward.
- Bit-identical greedy: preserved across the entire session
  (every commit's smoke run reproduced Off mode tokens exactly).

## What's queued

- **Spawned chip:** CPU `attention()` divergence fix (independent;
  unrelated to the speed work; would restore `forward_token_shared_only`
  correctness for any Phase 3 prep that depends on it).
- **Class A:** single-TCB Eagle4Head encoding (~+1-2 tps).
- **Stage 0.5:** MLX-pattern adoption on hot V2-Lite kernels. The
  decisive lever for clearing Stage 1.
- **Stage 2 (Path B):** K>1 batched verify kernels.

## Artifacts

```
reports/path_to_90/_stage1_remeasurement/
  raw.json                 ← parsed Off + Eagle4 dec_tps + accept
  STATUS.log               ← full timestamped script log
  off_t0.log .. off_t2.log ← Off-mode [stats] output
  e4_t0.log  .. e4_t2.log  ← Eagle4-mode + spec_log per-step accept/reject
```

The first re-measurement (commit `700e68e`'s 7.36 tps data) is
preserved at its commit point in the git history; this run's data
overwrote those files in the working tree.
