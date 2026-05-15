# Path-to-90 Stage 1 — A3 close: add_rmsnorm_f32 fusion — REJECTED at +3% gate

**Status:** HALTED. Infrastructure landed as opt-in `residual_fusion = "f32"`; default unchanged.
**Branch:** `claude/strange-proskuriakova-b5d48e`
**Base:** 2628a5c (post-Stage-3 audit).
**Date:** 2026-05-15

## Result

| Profile | dec_tps (trimmed-median) | Δ vs A4 baseline |
|---|---:|---:|
| A4 (current default) | 23.97 | — |
| **A3 (`residual_fusion=f32`)** | **22.70** | **−5.3%** |
| warm-10 median: A4 / A3 | 23.98 / 23.45 | −2.2% |
| min: A4 / A3 | 21.06 / 11.46 | −46% (cold-tail much worse) |

**Parity:** bit-identical 3-token greedy decode on all 12 baseline prompts. The 25 lib tests pass.

The kernel is semantically correct (parity confirmed). The regression is a GPU-side issue, not a correctness issue.

## Why it lost

Per-element parallelism collapsed. The unfused pair:

| Step | Unfused grid | Threads | Elements/thread |
|---|---|---:|---:|
| `add_inplace(x, addend, hidden=2048)` | `(8×256, 1, 1)` = 8 TGs | 2048 | **1** |
| `rmsnorm_f32(x → out)` | `(256, 1, 1)` = 1 TG | 256 | 8 |

The fused kernel runs in a single TG (256 threads) because the rmsnorm reduction needs a TG-local barrier — and cross-TG synchronization isn't available in Metal within one dispatch. So the add phase, which had 8× parallelism in the unfused version, now does 8 elements/thread sequentially in the single TG. The GPU compute cost of the add jumps roughly 8×; the small CPU dispatch-encode savings (~50 µs from one fewer dispatch × 24 fused pairs/token = ~1.2 ms/tok) get out-paid by ~2-3 ms/tok of extra GPU compute.

The fundamental tradeoff: race-safe single-TG fusion requires sacrificing per-element parallelism. For `hidden = 2048`, that's net-negative on M3 Pro.

## What might work (deferred)

To fuse without losing parallelism, the kernel would need either:
- **A two-pass design with a separate output buffer.** Pass 1: multi-TG `add` writes to `x_alt`. Pass 2: multi-TG `rmsnorm` reads `x_alt`, writes `out`. Still 2 dispatches — no win.
- **Atomic-buffer cross-TG sync.** Use an atomic counter to detect when all TGs have finished the add, then proceed to rmsnorm in TG0 only. Adds atomic ops + a spin-loop; defeats the purpose.
- **A different fusion target.** Fuse `add` with the *prior* kernel (the o_proj GEMV's write of `arena.out`) — i.e., the residual addend is written by the same kernel that produced it. Same multi-write-race issue.

The cleanest path is to leave `add_inplace` and `rmsnorm_f32` as separate dispatches and look elsewhere for engine wins.

## Pattern across recent rejections

A1 (flash-attn), A4.2 (moe-fc), A3 (add-rmsnorm) all rejected. The common thread: each lever predicted +3-6% based on dispatch-count or GPU-share heuristics. Each lost because:
- A1: tile-bookkeeping overhead unrecouped at short context (seq_len < FLASH_TG)
- A4.2: function constants gave compiler nothing new to fold (the inner loops were already small); register-allocator hit lower occupancy
- A3: single-TG fusion sacrifices per-element parallelism

**Post-A4/A5 the dominant lever space has shifted.** A5+A4 cleared the dispatch-overhead "easy money" identified in Stage 0. The remaining engine wins require either:
1. **GPU-quality improvements** — better kernel parallelism, occupancy, or memory layout. These need attribution-driven micro-bench work per-kernel, not from-the-plan heuristics.
2. **Footprint reduction** (Track B: KV quant, hot/cold expert tiering) — opens new headroom by changing what bytes get moved per token.
3. **Self-speculative decoding** (Track C: needs out-of-session training). Real ceiling-break.

## What stays in tree

- `add_rmsnorm_f32` kernel in `common.metal` — opt-in.
- `add_rmsnorm_metal_buf_tcb` dispatcher.
- `residual_fusion` profile field with default `"off"`.
- Engine helper `encode_add_and_rmsnorm_tcb` that routes based on the flag.
- `shader_hash` in the profile updated to `ee4a863548cbfd90d5f8b4b2` (the new combined source).
- Default behavior unchanged.

## Stage 1 cumulative (unchanged)

| Stage | dec_tps | Δ vs main |
|---|---:|---:|
| pristine main (v2.2.0) | 20.50 | — |
| A5 + A4 (shipped default) | **23.97** | **+16.9%** |
| A1, A4.2, A3 (rejected; opt-in) | — | — |

## Next per-session strategy

Three consecutive plan-order rejections suggests the original Stage-1 ordering needs re-grounding in attribution-driven evidence. Pulling forward from later plan stages:

- **Build B1 PPL harness + multi-prompt bench infrastructure**. Pure tooling. Unlocks Stage 2 (KV quant) quality gates and gives every future lever a more realistic measurement surface — many of the recent rejections may be context-length-sensitive (e.g., A1 should win at seq_len ≥ 1K but our bench's max seq_len is ~70).
- **Re-measure A1 / A3 on long-prompt workloads** after the harness lands. If A1 wins at chat-realistic context, it ships as a context-conditional schedule.
- **Defer Stage 1 micro-fusion until evidence-driven**. The trace already shows where time goes; the question is which remaining kernels respond to specialization (A4 did) vs which don't (A4.2, A1, A3).

This re-prioritization is documented in the next commit's report.
