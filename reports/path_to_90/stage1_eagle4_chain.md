# Path-to-90 step 10 — Stage 1 measurement

**Captured:** 2026-05-18 17:38–17:41 EDT
**Commit benched:** `5d0947e` (architecture closeout state)
**Host:** M3 Pro 18 GB
**Workload:** 3 prompts × 16-token greedy decode, `--speculate off` vs
`--speculate eagle4`, captured via
`tools/bench/stage1_eagle4_measurement.sh` (clean window: Claude.app
quit, slm idle).

## The numbers

dismantle's `[stats]` line gives decode-only `dec_tps` directly (the
wall-clock numbers the bench script computed include prefill, which
varies across runs from process cold-start; the canonical number is
`dec_tps` reported by the engine).

| prompt | mode | prefill_ms | decode_ms | **dec_tps** | draft accept |
|---|---|---:|---:|---:|---:|
| "The quick brown fox" | Off | 3833.6 | 589.8 | **27.13** | n/a |
| "Once upon a time" | Off | 1264.8 | 590.4 | **27.10** | n/a |
| "def fibonacci(n):" | Off | 1924.6 | 594.0 | **26.94** | n/a |
| "The quick brown fox" | Eagle4 | 6245.1 | 30783.7 | **0.52** | 0/16 |
| "Once upon a time" | Eagle4 | 4407.8 | 29602.7 | **0.54** | 0/16 |
| "def fibonacci(n):" | Eagle4 | 5972.8 | 16601.6 | **0.96** | 1/15 |

**Median Off dec_tps: 27.10** (matches Stage 0 baseline 26.93 — fully
consistent. Off mode is unchanged by step 8's wire-up.)

**Median Eagle4 dec_tps: 0.54.** That's a 50× regression vs Off.

**Cumulative draft acceptance: 1/47 = 2.13 %**. Eagle4 head was
trained to 87.48 % target-argmax acceptance on MLX bf16 V2-Lite forwards;
seeing 2 % on dismantle-CPU-walk inputs confirms the CPU attention()
divergence (foundation_halt.md) makes the head's predictions
essentially uncorrelated with V2-Lite's argmax. The bit-identicality of
emission is preserved (Off and Eagle4 emit the same tokens — same
`33747, 855, 254, 24547, 5025, 5025, ...` repetition collapse) because
emission goes through the GPU verifier path; only the draft is wrong.

## Block-ship gate

Execution plan § Stage 1 block-ship: **18–24 tok/s ± with zero quality
regression on Spec-Bench MT-Bench**.

| | required | observed |
|---|---:|---:|
| Eagle4 dec_tps | ≥ 18 | **0.54** |
| Quality regression vs Off | zero | **zero** (bit-identical greedy at 16 tokens, all 3 prompts) |

**Gate: HALT at speed.** Quality side passes; speed side fails by ~33×.

Per execution_plan.md § Stage 1: "If step 10 lands above 20 tok/s,
push through to Stage 2 first. If it lands below 15, the regression is
severe enough that the kernel-efficiency dividend matters MORE than
the spec-decode landing, and Stage 0.5 jumps the queue."

We're at 0.54, an order of magnitude below the 15 threshold. The
reinject rule fires.

## What the halt is pointing at

It is NOT pointing at Stage 0.5 MLX-pattern adoption (the literal
plan branch for < 15 tok/s). Reason: Off mode runs at the expected
27 tok/s — V2-Lite's own forward isn't the bottleneck. The Eagle4
mode's 0.54 tok/s comes from the CPU-walk eagle4 capture path,
which is dismantle-specific scaffolding, NOT V2-Lite's hot kernel
inefficiency.

The architectural unlock is the same item already identified in
`reports/path_to_90/architecture_closeout.md § Architectural
followups #2`: **GPU-side eagle4 capture**. Instrument the Wedge C
TCB path to extract `x_buf` at layers 2/13/25 and call
`ffn_shared_only` at layer 26 GPU-side, eliminating the CPU walk
entirely. Expected outcome: Eagle4 decode tps lands at ~22–25
tok/s (slightly below Off due to head-call overhead at ~5 ms; the
plan's 12–22 tok/s band captures this).

Once GPU-side capture lands AND the CPU attention() chip resolves,
the eagle4 head's draft accuracy should recover to ~87 % (the
trained number). Then Stage 1 = Off baseline minus a small head-
call tax, Stage 2 needs Path B kernels for actual amortized
speedup over Off.

Stage 0.5 MLX-pattern adoption (the literal "< 15 tok/s → reinject"
branch) is still mandatory per step 2's decision, but it's a
PARALLEL track that doesn't gate Stage 1's correctness. It gates
the eventual Stage 5 headline number.

## Decision

1. **Do NOT proceed to Stage 1 ship.** Speed gate failed by 33×.
2. **Next architectural unlock: GPU-side eagle4 capture** (~half-day
   focused Metal work). Re-run step 10 measurement after.
3. **In parallel:** the spawned chip for the CPU attention()
   divergence fix — independent of capture path, restores eagle4
   stats to meaningful values.
4. **In parallel:** Stage 0.5 MLX-pattern adoption work on
   gemv_q4_k_v3, MoE pair matmul, MLA decode (per step 2 decision).
   Lands the eventual headline number; doesn't affect Stage 1.

Step 11 (routing-recall fine-tune) should wait until #1 + #2 resolve
— there's no point fine-tuning a head whose inputs come from a
broken CPU walk that we're about to replace with GPU-side capture
anyway.

## Methodology notes (for future bench scripts)

The bench script's wall-clock `tps = tokens / wall_time` is contaminated
by process startup + prefill. dismantle prints decode-only `dec_tps`
in its `[stats]` line; future scripts should grep that out instead.
Off wall-clock was 2.4–4.6 tps; Off decode-only is 26.9–27.1 tps
(an order of magnitude difference, dominated by ~2–4 s of per-process
model load + prefill for short 4-token prompts).

I'll fix `tools/bench/stage1_eagle4_measurement.sh` to parse the
`dec_tps=` field on the next iteration so the raw.json's `median_tps`
matches the architecturally-meaningful number.

## Artifacts

```
reports/path_to_90/_stage1_capture/
  raw.json                 ← bench-script-computed (wall-clock; see note above)
  STATUS.log               ← full timestamped script log
  off_t0.log .. off_t2.log ← `dismantle generate` Off-mode output incl. [stats]
  e4_t0.log  .. e4_t2.log  ← Eagle4-mode output incl. per-step accept/reject
```

Heavy / regeneratable; all under `_stage1_capture/`. Committed
selectively (this report + raw.json + the small log files for the
audit trail).
