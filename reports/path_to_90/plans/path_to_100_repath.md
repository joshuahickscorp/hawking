# Path-to-100 repath — dual-track post-clean-bench

**Status:** REPATH. Replaces the post-F.2 L7-only sequence in [path_to_100_retool.md](path_to_100_retool.md).
**Trigger:** Clean-window bench `reports/path_to_90/_bench_20260520T143008/` reveals all speculative-decode paths are regressions vs off-mode, breaking the assumption that L7 + spec-decode stack additively toward 100.
**Realistic confidence:** LOW that 100 ships. MEDIUM that 60-70 is reachable. The plan is dual-track because no single track gets there.

## Ground truth (clean-window, 2026-05-20)

```
config                            median  dec_tps  vs off
off / sequential / K=1             26.87   ─── baseline ───
ngram / parallel-k / K=1           26.48   −1.5%
eagle4 / sequential / K=1          18.01   −33%
eagle4 / parallel-k / K=4           7.52   −72%
```

Three things this forces:

1. **Off-mode is the ceiling.** Every speculative-decode path here is a regression.
2. **Eagle4 K=1 is 33% slower than off** despite being defined as "emit verifier's argmax + run head for stats" — bit-identical to off by construction (see [deepseek_v2.rs:1462-1463](crates/dismantle-core/src/model/deepseek_v2.rs:1462)). The 8.9-tps gap is pure capture-path tax with zero acceptance benefit.
3. **Eagle4 chain K=4 is 72% slower than off** — `K=4 × verifier_cost` per outer iter dominates, and acceptance isn't recovering it.

## The math of why we need both tracks

To reach 100 dec_tps from 26.87:
- Need **3.7× multiplier** total
- Kernel ceiling per [v110_path30_findings] memory: ~35-40 tps (LM head is only 4% of forward, MoE v3 keeps regressing, f16 residual breaks at depth)
- So kernel track alone caps at ~**1.5×**
- Remaining gap requires **2.5× spec-decode multiplier on top of accelerated off** — equivalent to chain K=4 going from 7.5 → ~87+

Either track alone misses 100 by a lot. Both tracks landing puts 100 in reach (70-105 envelope).

## Track 1 — off-mode kernel acceleration

**Target:** 26.87 → 35 dec_tps (1.3×). The verifier-path acceleration story.

**Why it's defensible:** Off-mode is the highest-tps path right now. Every kernel here also accelerates the verifier inside every speculative-decode mode, so the win compounds with Track 2.

**Levers:**
| Lever | File | Status |
|---|---|---|
| L7.D — `moe_batched_gemm_q4_indexed_v3` | [phase_l7d_plan.md](phase_l7d_plan.md) | Architected, MLX ref now in tree at [mlx_lm_ref/](../mlx_lm_ref/) |
| L7.E — `gemv_q4_k_v3_mlx` standalone GEMV | (same MLX ref) | Architected, blocked on L7.D learnings |
| Attention path (Phase 2 + 3 fusion) | TBD | Unscoped — needs profile trace to confirm there's headroom |
| LM head — already at 4% of time per memory | n/a | Not a target |

**Sequencing:**
1. Read [mlx_lm_ref/quantized.h:749-815](../mlx_lm_ref/quantized.h) (qmv_fast_impl). Identify divergence from `gemm_q4_k_m_v3_xtg_sumy`.
2. Update [phase_l7d_plan.md](phase_l7d_plan.md) with the actual MLX inner-block (replacing hypothesis).
3. Implement → parity → clean-window A/B bench → wire if ≥5% off-mode improvement.
4. Same flow for L7.E.

**Per-lever bench gate:** ≥5% off-mode dec_tps improvement. Below that = ship the kernel as dormant, document, move on.

**Track 1 verdict at +0%:** if no L7 lever lands a clean-window win, Track 1 is dead. Off-mode caps near 27. Path-to-100 is over. Move to Track 2 exclusively or stop chasing 100.

## Track 2 — speculative-decode recovery

**Target:** chain K=4 from 7.52 → 50+ dec_tps. This is the multiplier track.

**Why it has to be its own track:** Currently spec-decode loses to off-mode at every K. Without diagnosis, "fixing" is unfocused. The track is structured as a forensic sequence, not implementation.

**Step 2A — diagnose the eagle4-K=1 tax (8.9 tps).**

At K=1, eagle4 sequential should emit v2_argmax bit-identically to off and then run head propose for stats. The head propose is dispatched on either Metal or AMX per `EAGLE4_BACKEND`. The 33% slowdown vs off means the eagle4-specific work (capture + head propose) is costing 8.9 tps per token with zero acceptance benefit.

Sub-steps:
1. Re-run bench with `EAGLE4_BACKEND=metal` and `=cpu` to isolate the head-forward cost from the capture cost.
2. Add `DISMANTLE_SPEC_LOG=1` to one prompt at K=1, capture per-step timing breakdown via the `[spec/eagle4-chain]` log line (or add one to the K=1 path if absent — the `chain_k >= 2` branch has it; K=1 path may not).
3. Compute: tax = (off_us_per_token) - (eagle4_K1_us_per_token). Allocate that tax across {capture forward, h_shared compute, head propose, head argmax}.
4. **Gate:** if tax is dominated by `forward_token_eagle4_capture` (one of the four), it's an instrumentation cost — could ship `eagle4_stats_off` flag that skips capture when stats aren't needed. Defines an immediate +30% K=1 win.
5. **Gate:** if tax is dominated by head propose, then EVERY chain step pays this cost — propose is the bottleneck.

**Step 2B — diagnose chain-K=4's regression.**

Off-mode runs 1 verifier per token = 26.87 tps. Chain K=4 runs 1 verifier (K+1 batched) + K head propose + accept-or-reject per outer iter. Outer iter emits ~1-K+1 tokens depending on acceptance.

The math floor for chain K=4 to beat off:
```
chain_dec_tps = (1 + accept) / outer_step_seconds
must be ≥ 26.87 to break even with off

If outer_step ≈ off_step × 1.5 (verifier is K-batched, K head proposes amortize):
chain_dec_tps = (1 + accept) / (1.5 × 1/26.87) = 17.9 × (1 + accept)
- accept=0 → 17.9 tps (currently 7.5, so we're 58% under the no-accept floor)
- accept=0.5 → 26.9 tps (break even)
- accept=2.0 → 53.7 tps (path-to-50)
- accept=3.0 → 71.6 tps (path-to-70)
- accept=4.0 → 89.6 tps (path-to-90 ← original phase target)
```

Current 7.5 tps with K=4 means either:
- outer_step is far more than 1.5× off_step (most likely — eagle4 capture tax + K=4 verifier overhead + K head propose overhead all stack)
- OR acceptance is effectively zero (head almost never predicts correctly)
- OR both

Sub-steps:
1. Capture spec_log at K=4 over 3 prompts; record per-step `accept` count distribution.
2. **Gate:** if mean acceptance < 0.5 over K=4, the head isn't good enough — F.3 (Rust port of medusa head) or F.5 (hybrid tree) or new draft head training is needed. Not a kernel problem.
3. **Gate:** if acceptance is fine (≥1.0) but outer_step is bloated, profile the chain loop. Per-step breakdown should pinpoint whether the cost is in propose, verifier, or the host-side glue.

**Track 2 verdict at "head isn't predicting":** path to 100 dies at this gate. F.2 already proved medusa K=8 doesn't converge to acceptance; if the current head's K=4 acceptance is also weak, draft-head quality is the architectural wall. Document and stop chasing 100.

**Track 2 verdict at "head is fine but glue is slow":** L5 Lever A/B work becomes relevant again, plus argbuf rollup, persistent threads for chain steps, etc. The implementation-heavy path.

## Sequencing recommendation

```
Session N+1 (next, ~3-4 hr):
  - Step 2A diagnosis: re-bench with EAGLE4_BACKEND={metal,cpu}, spec_log K=1
  - Result analysis: locate the 8.9-tps tax
  - If tax is recoverable (instrumentation): ship the fix (eagle4_stats_off flag, ~30 LoC)
  - If tax is architectural: document, move to Step 2B

Session N+2 (~3-4 hr):
  - Step 2B diagnosis: spec_log K=4, acceptance distribution
  - Decision gate: is the head good enough?
    YES → Track 2 implementation work (L5 Lever B, argbuf rollup, ...)
    NO  → write path_to_100_dead.md, scope back to path-to-60/70

Session N+3+ (parallel track, dependencies satisfied):
  - L7.D inner-block implementation per MLX ref
  - Parity → clean-window A/B → wire only on ≥5% off win
  - Same flow for L7.E
```

## Honest failure modes

1. **Off-mode kernel work plateaus at 27-30** — already happened with v1.1.0's "stuck" finding. If L7.D/L7.E don't deliver, kernel ceiling is the wall.
2. **Eagle4 K=1 tax is architectural** (the capture buffer is required for stats AND for chain seed; can't be cleanly disabled). Then chain decode inherits the tax permanently.
3. **Draft head acceptance is weak** at chain K=4. F.2 ruled out one approach (medusa K=8); the current eagle4_v3 head may be similarly capped. Then no spec-decode multiplier exists at this model size.
4. **Apple GPU just can't go faster at single-stream decode.** MLX-LM benches on DeepSeek-V2-Lite Q4_K_M peak around 30-35 tps on M3 Pro. We may be near the silicon ceiling for batch=1.

Any one of these kills 100. Two of them are already partially confirmed (1 from memory, 4 from external benches). The dual-track plan budgets ~2 weeks of focused work before the picture clarifies enough to call it.

## What this commit does

- Replaces "path-to-100 = L7 kernel work" framing with "path-to-100 = kernel + spec-decode recovery, both required."
- Makes spec-decode recovery a *diagnostic* track first, not an implementation track. We don't know where the regression is yet.
- Adds explicit verdict-gates at each diagnostic step so we stop chasing 100 when the evidence says we can't reach it.
- Doesn't kill L7.D / L7.E (still relevant for off-mode acceleration), just demotes them from "stack to 100" to "ship off-mode wins regardless."

The 60-70 dec_tps ceiling from [post-F.2 retool] still stands as the realistic outcome. 100 requires every gate to pass; we're committing to know which gate fails fast, not committing to the 100 number.
