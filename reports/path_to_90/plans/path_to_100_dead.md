# Path-to-100 — DEAD

**Status:** CLOSED. Supersedes [path_to_100_repath.md](path_to_100_repath.md). Path-to-100 via speculative decode is architecturally blocked; Track 1 (off-mode kernel acceleration) alone caps at ~35-40 dec_tps per v110_path30_findings memory.
**Trigger:** Step 2B clean-window bench 2026-05-20 16:27/16:28. Both K=4 and K=8 mean_accept fall below the no-accept floor; bigger K makes things strictly worse.
**Realistic ceiling:** **path-to-30 to path-to-40** via Track 1 kernel work. 50-70 envelope requires draft-head retraining (out of scope).

## What killed it

Clean-window measurement at K=4 (`reports/path_to_90/_bench_step2b_20260520T162700/`):

```
accept=0/4   62/84   73.8%
accept=1/4   22/84   26.2%
accept=2/4    0/84    0.0%
accept=3/4    0/84    0.0%
accept=4/4    0/84    0.0%

mean_accept            = 0.262
median_outer_step_ms   = 258.00  (6.93× off_step_ms = 37.24)
acceptance for break-even at K=4 = 5.93   (impossible: max is 4)
chain dec_tps          = ~7-10   (vs off=26.85)
```

Clean-window at K=8 (`_bench_step2b_20260520T162836/`):

```
accept=0/8   62/82   75.6%
accept=1/8   15/82   18.3%
accept=2/8    5/82    6.1%
accept=3..8/8  0/82    0.0%

mean_accept            = 0.305
median_outer_step_ms   = 441.40  (11.82× off_step_ms = 37.35)
acceptance for break-even at K=8 = 10.82
chain dec_tps          = ~4-5    (regressing further)
```

**Three facts that close the lever:**

1. **Per-token acceptance saturates at ~26%.** The head gets the first draft right ~1 in 4 tokens. After the first reject, the rest of the chain is dead (conditional probability of accepting subsequent drafts is near zero).

2. **Going deeper hurts.** K=8 vs K=4: outer_step grows 1.71× (linear in K), mean_accept grows only 1.16× (saturating). Step inflation grows faster than acceptance — bigger K is strictly worse.

3. **Step inflation is ALREADY too high at K=4** (6.93× off_step). Even if mean_accept were a magical 5.0/4 (impossible), chain_tps = 6/258ms = 23.3 tps — still under off=26.85. The host-side glue + K-batched verify cost makes the chain mode structurally uncompetitive.

The conclusion is **independent** of L5 Lever B (chain-step pipelining) and argbuf rollup work — those reduce step_inflation, but at acceptance=0.26 there's no realistic step_inflation that wins. Break-even at acceptance=0.26 requires step_inflation ≤ 1.26 — chain mode at K=4 would need to be ~36 ms per outer iter (≈ off_step). That's slower than off by definition; chain inherits the full off forward as its verifier.

## Why this matches F.2's NEGATIVE result

[phase_f2_negative.md](../closeouts/phase_f2_negative.md) ruled out medusa K=8 at 2025-05-19: per-head top-1 acceptance plateaued at ~5% across heads, top-10 at ~30%. The eagle4_v3 head was trained differently but lands at the same per-token acceptance ceiling (~26%). Two independently trained draft heads on this model converge to similar architectural ceilings → the **bottleneck is the model/data, not the head architecture**.

DeepSeek-V2-Lite Q4_K_M is too easy of a target — at the per-token entropy level required for speculative decode to win, the verifier's own argmax distribution is too sharply peaked for the draft head to win meaningful coin flips beyond the first.

## What survives

**Track 1 — off-mode kernel acceleration.** Still relevant. Sources:
- L7.D (`moe_batched_gemm_q4_indexed_v3`, MLX ref in tree) — primary target
- L7.E (`gemv_q4_k_v3_mlx`) — secondary
- Attention path Phase 2+3 fusion if profile reveals headroom

Per [v110_path30_findings memory](../../memory/v110_path30_findings.md): off-mode caps near 35-40 tps. Even at the upper bound, that's 1.3-1.5× the current 26.87 baseline. **Path-to-100 from a 27-tps base requires a 3.7× multiplier that cannot come from kernel work alone.**

**Realistic post-Step-2B target: path-to-30 to path-to-40.**
- Floor: 30 tps (1.12× off baseline; 5% L7 win that's achievable)
- Ceiling: 40 tps (kernel ceiling per memory; requires both L7.D and L7.E to land 5%+)
- Stretch: 50 tps (requires unforeseen kernel headroom + attention-fusion wins)

100 tps requires either:
1. A new draft head architecture trained on harder targets (chain acceptance ≥ 2.0/K) — multi-week project, out of current scope
2. A different model (larger Q at FP16, or a draft-model-natural like Llama-7B with EAGLE-3 distillation) — out of scope
3. Apple Silicon ceiling break (M5 / GPU dispatch model overhaul) — out of our hands

## Out of scope, going forward

- Any further chain-K acceptance probing (K=2 with current head would land 7-12 tps, same wall)
- L5 Lever B (chain-step pipelining) — would only help if Gate 2 had fired, which it didn't
- L5 Lever A wiring (eagle4_rmsnorm_residual_gate) — kernel exists but +1-2 tps gain doesn't move the needle
- argbuf rollup, persistent threads, K-batched verifier amortization — all targets a chain path that loses to off anyway

## What unblocks more later (attended)

- **Head retraining.** F.3 medusa Rust port + new training run with longer-context targets. Would need draft acceptance distribution check before integration (don't repeat F.2 / Step 2B's mistake of integrating a head that hasn't been validated for chain-K acceptance).
- **Hybrid tree (F.5).** Tree-attention speculative decode (Medusa-2, EAGLE-3 tree). More invasive but theoretically higher acceptance at fixed branching width.
- **Different verifier model.** Try Qwen2.5-Coder-1.5B or similar; the smaller model + sharper per-token entropy may make speculative decode beneficial at a smaller absolute floor.

## What this commit does

- Closes the spec-decode track decisively. No more "what if K=2"-style probing — the data shows acceptance saturates at ~26% across K, so K=2 lands the same place at slightly better step inflation but still loses to off.
- Demotes the path-to-100 target. The new realistic target is **path-to-30/40**; the 60-70 envelope from [path_to_100_repath.md] required the chain track to land, which it didn't.
- Keeps Track 1 (off-mode kernel acceleration) alive as the sole remaining knob.
- Suggests the head retraining track only as future attended work; not committing to it.

The dual-track plan budgeted 2 weeks before the picture clarified. It clarified at session 2 — Step 2A showed K=1 tax is architectural, Step 2B showed chain-K acceptance is architectural. Both pillars fall. **Path-to-100 ships when (a) a viable draft head with mean_accept ≥ 2/K at K=4 exists AND (b) Track 1 lands ≥30% off-mode improvement.** Neither is in this branch's scope.
