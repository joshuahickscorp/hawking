# Step 2B — NEGATIVE

**Halted at:** 2026-05-20 16:28 EDT
**Halted on:** Gate 1 — draft head architectural acceptance wall

## Root cause

The eagle4_v3 draft head's per-token acceptance saturates at ~26% across all chain depths tested (K=4 and K=8). At K=4, mean_accept=0.262 with 74% of outer iters scoring accept=0; at K=8, mean_accept=0.305 with 76% accept=0 and 0% reaching accept≥3. Bigger K is strictly worse because step_inflation grows linearly while acceptance saturates. The per-token acceptance ceiling is the wall — chain depth optimizations cannot recover it.

## What ran

Clean-window benches (Cmd-Q Claude) via tools/bench/path_to_100_step2b.sh:

| run | K | mean_accept | median_step_ms | step_inflation | chain_dec_tps | output |
|---|---|---|---|---|---|---|
| 162700 | 4 | 0.262 | 258.00 | 6.93× off | 7-10 | reports/path_to_90/_bench_step2b_20260520T162700/ |
| 162836 | 8 | 0.305 | 441.40 | 11.82× off | 4-5  | reports/path_to_90/_bench_step2b_20260520T162836/ |

off baseline: 26.85 dec_tps (37.24 ms/token).

Accept distributions captured to `chain_steps.csv` per run. Full chain log preserved in `chain_log.txt`. Per [path_to_100_dead.md](../plans/path_to_100_dead.md): both K=4 and K=8 fail Gate 1 (mean_accept < 0.5). The brief's "head isn't predicting" verdict gate has fired.

## What attended work unblocks

This is a closure, not a temporary block. The path-to-100 target dies at this gate per the [path_to_100_repath.md](../plans/path_to_100_repath.md) sequencing recommendation:

> Session N+2 (~3-4 hr):
>   - Step 2B diagnosis: spec_log K=4, acceptance distribution
>   - Decision gate: is the head good enough?
>     NO  → write path_to_100_dead.md, scope back to path-to-60/70

(Adjusted: realistic ceiling per Track 1 alone is path-to-30/40, not 60/70 — Track 2 was load-bearing for the upper envelope.)

To revisit, attended work needed:
1. **Draft head retraining.** Fresh head with a training distribution that targets chain-K acceptance ≥ 2/K. F.3 medusa Rust port or new eagle4 variant. Pre-integration gate: synthetic K=4 acceptance check on held-out prompts before any bench integration. Don't repeat the F.2 / eagle4_v3 mistake of shipping a head that hasn't been chain-K-validated.
2. **Hybrid tree (F.5)** if straight chain remains capped. Tree-attention reuses the same per-token entropy differently — different accept-or-reject math may surface acceptance the chain mode can't.
3. **Different verifier model** — if DeepSeek-V2-Lite Q4_K_M is itself the wall (the verifier's per-token entropy is too peaked for any draft head to win meaningful coin flips beyond the first), retry on Qwen2.5-Coder-1.5B Q4_K_M or similar.

## Followups

- Update [path_to_100_repath.md](../plans/path_to_100_repath.md)'s status to "SUPERSEDED by path_to_100_dead.md" (attended — not autonomously modifying a plan doc).
- Update auto-memory ([MEMORY.md](../../../memory/MEMORY.md)) `path-to-100 repath` entry to point at the new dead doc (attended).
- Track 1 (off-mode kernel acceleration) becomes the sole remaining lever for path-to-30/40. Sequencing: L7.D inner-block per MLX ref → parity → clean-window A/B → wire if ≥5% off win; same flow for L7.E. Per-lever bench gate at +5% off-mode dec_tps.
- No further chain-K probing this branch. K=2 with current head lands ~7-12 tps (per the math: lower step_inflation than K=4 but same acceptance ceiling) — still loses to off. No need to verify.

## Anchoring evidence

- K=4 chain log shows the head's repetition pathology directly: for prompt "The quick brown fox", iters 2+ emit "dog dog dog" patterns. The head accepts 1/4 when the model has fallen into a repetition trap (predicting the obvious repeat) and 0/4 otherwise. This isn't a calibration issue or a temperature artifact — it's the head not generalizing beyond trivial pattern continuation.
- K=8's 6.1% accept=2 rate is the most acceptance the head ever delivered. 0% above that confirms there's no useful tail.
- step_inflation linearity (6.93× → 11.82× as K goes 4→8) confirms the chain-loop cost is dominated by K-scaling components (head proposes + K-batched verifier). Step 2A's per-phase data implied this; Step 2B's K-sweep validates.
