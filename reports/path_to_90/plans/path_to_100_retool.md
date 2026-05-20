# Path-to-100 retool — realistic target replacing path-to-150

**Date:** 2026-05-19
**Supersedes:** path-to-150 trajectory (150 dec_tps target).
**New target:** 100 dec_tps. Better than llama.cpp on M3 Pro and within
reach of currently-planned levers.

## Why retool from 150 → 100

Phase E (tree decode, +30–50 dec_tps) was killed by E.0.a's kbatch
K-sweep bench (commit `c84671a`). That removed the only lever that
could carry the final +30–50 from a 100-ish baseline to 150. Without
E, the remaining planned levers (F + L7 + L5) sum to a best case of
~+78 dec_tps over the 26.78 baseline → ceiling ~105.

The 150 stretch target now requires F.2 to substantially overshoot
published medusa (>2× chain on real outputs). The F.2 subset epoch-0
numbers (head 0 60% top1, late heads 4–11%) suggest realized effective
K of 3–5 across the chain, which lands toward the middle of the
+20–40 band. That's a +30 realistic outcome, not a >+50 overshoot.

100 is the realistic horizon. It still beats llama.cpp on M3 Pro and
gives the project a defendable ship target.

## Updated trajectory math

| Lever | Best | Realistic | Effort | Status |
| --- | ---: | ---: | --- | --- |
| Baseline | 26.78 | 26.78 | — | shipped |
| F.2 (medusa training) | +40 | +30 | 1 night | running now |
| F.3 (Rust port) | (gates F.2) | (gates F.2) | 3–5 days | queued |
| L7 (kernel rewrites) | +30 | +20 | 2–3 days | plan ready |
| L5 (chain pipeline) | +8 | +5 | 3–6 hr | plan ready |
| **Subtotal F + L7 + L5** | **+78** | **+55** | | |
| **Projected dec_tps** | **105** | **82** | | |
| F.5 (hybrid tree-of-medusa) | +10 | +6 | 1–2 weeks | stretch, gated on F.3 |
| Q5_0 paired-nibble (memory `v230_t215_close.md`) | +5 | +3 | 1 day | stretch, untested |
| **Stretch ceiling** | **+93** | **+64** | | |
| **Projected dec_tps w/ stretch** | **120** | **91** | | |

Hitting 100: needs F + L7 at upper-band of realistic OR the stretch
levers landing. Hitting 90: needs F + L7 at realistic mid-band. Hitting
60: floor case with F alone.

## Phase sequence (revised)

1. **F.2** — medusa training (running tonight)
2. **F.3** — Rust port of medusa heads (3–5 days; gated on F.2 acceptance)
3. **L7** — kernel rewrites (L7.2 fusion + MLX Q4_K_M GEMV); 2–3 days
4. **L5** — chain-decode pipeline restructure; 3–6 hr
5. **(Stretch)** F.5 hybrid tree-of-medusa once F.3 ships; gated on
   whether F.2 + L7 + L5 already cleared 90 dec_tps
6. **(Stretch)** Q5_0 paired-nibble fix #3 (untested per memory)

## Killed phases (do not pursue)

- **Phase E** (tree decode, both v1 and v2 plans). E.0.a gate negative.
  MLA kbatch kernel cannot scale past K=4 on M3 Pro.
- **path-to-150 stretch lever search.** All four tier-3 audit levers
  (ICB, megakernel, two of three Q5_0 fixes) ruled out. Per memory
  `v230_icb_dead.md`. No remaining hidden levers in the published
  inference-optimization literature applicable to M3 Pro + DeepSeek-V2-Lite.

## Methodology each phase inherits

Every phase from F.3 forward MUST apply patterns from BOTH
`acceleration_patterns.md` (1–10) AND `methodology_distilled_post_f2.md`
(11–20). The pre-launch checklist at the bottom of those docs is the
gating bar.

**Most-impactful for time savings on remaining phases:**

- **Pattern 11 (capture-then-iterate):** F.3 retrains use the same
  shards F.2 trained on; budget seconds for any re-projection, not
  hours.
- **Pattern 12 (frozen-weight npz):** F.3 trainer runs concurrent with
  L7 kernel benches. No clean-window blocker.
- **Pattern 13 (smoke → subset → full):** F.3 retrain runs a 30-step
  smoke before any meaningful training time.
- **Pattern 16 (front-load levers):** L7 infra commit bakes all kernel
  patterns (xtg + L7.2 fusion + MLX Q4_K_M + MoE Q4 indexed v3) into
  one shippable infra; then iterate parity + bench.
- **Pattern 18 (hyperparams from first principles):** L7 kernel
  geometry (ROWS_PER_TG, simdgroup count, threadgroup memory layout)
  is reasoned from M3 Pro arch, not grid-searched.
- **Pattern 20 (convergence-based epoch cap):** F.3 retrain inherits
  F.2's observed convergence epoch + 20% safety. If F.2 converges at
  epoch 6, F.3 caps at epoch 7. Cuts ~40% wall-clock.

## Acceptance bar for "path-to-100 done"

- 100 dec_tps demonstrated in a clean-window bench (Claude quit, no
  concurrent slm training, ≥10 trial median)
- Per-component dec_tps deltas attributed (F vs L7 vs L5) in a
  closeout summary
- Parity gates green: path_b_parity, eagle4_decode_parity, eagle4_chain_pipeline_smoke

If 100 is met → ship as v2.4.0 (next minor after v2.3.0). If 100 is
not met but 80+ is, ship as v2.4.0-rc with a clear "stretch to 100
needs F.5 or Q5_0 paired-nibble" callout in the release notes.

## Note on bench contamination

Per memory `bench_contamination.md`, Claude Code session inflates
dec_tps 4–5×. Every reported number above is a CLEAN-WINDOW number.
In-loop wedges validate via kernel-count ratios + parity (not raw
TPS). All path-to-100 acceptance is gated on clean-window benches.
