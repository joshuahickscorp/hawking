# Oracle — L2.2 co-activation permutation (kill-or-keep)

**Date:** 2026-05-30  
**Lever:** bible §8.1 L2.2 — contextual sparsity via offline co-activation permutation (resurrection of dead block-256).  
**Verdict:** **NO-GO**  

Best post-permute skippable 1.5% (block 32) is below the ~30% bar. Co-activation permutation does not recover enough contiguous mass — the lever re-dies, consistent with SwiGLU lacking ReLU's hard zeros.

## Method

- Capture `_capture/q3b_ffn.bin` stores only per-256-block reductions, so per-neuron data was **reconstructed**: dequantized the Q4_K gate/up weights from `models/qwen2.5-3b-instruct-q4_k_m.gguf` and computed the true SwiGLU activation `a = silu(gate·x)·(up·x)` (F=11008) for each captured token, x = captured `norm_in` (hidden=2048).
- **Reconstruction validated** against captured blockmax/blockl2: median relative L2 error per 256-block = **0.0000** (PASS <0.05).
- Permutation = average-linkage hierarchical clustering leaf order on the neuron co-firing correlation matrix (cold neurons parked contiguously at the tail). This is a best-case OFFLINE static reorder (upper bound vs any learned predictor).
- Byte-cut metric = drop whole blocks (smallest L2 energy first) while keeping >= 99% of activation L2 energy (same oracle as the Track-B block-256 gate). Mean over 5 sampled layers [0, 9, 18, 27, 35], up to 400 tokens/layer.

- Mean active neurons/token (|a|>0.0001): **99.3%**  
- Mean permanently-cold neurons: **0.0%**

## Skippable block fraction @ 99% recall (pre → post permute)

| block | skippable PRE | skippable POST | byte-cut POST (×0.72 FFN) |
|------:|--------------:|---------------:|--------------------------:|
| 32 | 0.3% | 1.5% | 1.1% |
| 64 | 0.1% | 0.8% | 0.6% |
| 128 | 0.0% | 0.3% | 0.2% |

### Per-layer detail

| layer | act/tok | dead | B32 pre→post | B64 pre→post | B128 pre→post |
|------:|--------:|-----:|----------:|----------:|----------:|
| 0 | 98.2% | 0.0% | 0.1%→2.8% | 0.0%→1.6% | 0.0%→0.8% |
| 9 | 99.3% | 0.0% | 0.1%→2.2% | 0.0%→1.3% | 0.0%→0.3% |
| 18 | 99.6% | 0.0% | 0.0%→0.4% | 0.0%→0.1% | 0.0%→0.0% |
| 27 | 99.7% | 0.0% | 0.0%→0.4% | 0.0%→0.1% | 0.0%→0.1% |
| 35 | 99.9% | 0.0% | 1.2%→1.6% | 0.6%→0.8% | 0.1%→0.2% |

## Comparison to the dead block-256 result

- Prior (`reports/dead_levers.md`): block-256 oracle skippable = **0.2%** @99% recall; active neurons scattered (~5.6 active channels/256-block, ~2.2%), participation-ratio sparse but granularity-mismatched. Block-256 predictor declared DEAD.
- This oracle adds (a) finer blocks {32,64,128} and (b) an offline co-activation **permutation** to cluster the scattered active neurons into contiguous runs.
- Result: even at block 32 with the best-case permutation, skippable goes 0.3% → 1.5%. The permutation moves the number only marginally: SwiGLU activations are dense-and-small, not hard-zero, so there is little all-cold contiguous mass to gather even after reordering. Same root cause as block-256, confirmed at finer granularity.

## Why permutation cannot help here (two confirming diagnostics)

Permutation needs (a) a small active set and (b) a *stable* active set across tokens. Neither holds for q3b's SwiGLU:

- **No energy concentration.** Keeping 99% of FFN L2 energy requires **39% (L0) to 53% (L18) of all 11008 neurons** per token (4318–5831 neurons); even 90% recall needs 9–20%. Effective active count (participation ratio L1²/L2²) is 15–30% of neurons. Energy is broadly spread, so most blocks carry non-negligible mass regardless of layout — there is no large all-cold mass to skip.
- **No active-set stability.** The top-200 highest-energy neurons overlap only **~22–25% (Jaccard) token-to-token** — the active set is strongly input-dependent. A *single static* offline permutation cannot pack into contiguous blocks a set that reshuffles every token. (This is exactly the input-dependence risk the bible flagged for L2.2.)

## Honest caveats

- These are best-case ORACLE numbers (a single fixed offline permutation + an omniscient per-token block selector). A deployable runtime needs a cheap PREDICTOR of the active set; it is strictly worse. So the oracle is an upper bound: a NO-GO here is decisive; a GO would still need a predictor oracle.
- Recall is measured on the FFN intermediate L2 energy (the down-projection input), not end-task quality; 99% energy recall is the same proxy the Track-B gate used.

