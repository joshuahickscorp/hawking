# Sub-bit closure ledger — Qwen3-235B-A22B-Instruct-2507

Hard ceiling: `complete_bits / original_weight_count <= 1/1`, enforced by
`tools/foundry/one_bit_ceiling.assert_complete_bpw_le_one` in exact rational arithmetic.
No arm above 1.0 is scheduled in this campaign, and none may be.

## Method families run in this generation

| Lane | Method | Status | Evidence |
|---|---|---|---|
| A | M01′ scale-invariant VQ | **implemented, measured on real weights** | `M01_FUNCTION_AWARE_PROBE.json` |
| A | M02 activation-aware (diag-Hessian) fitting | implemented, machinery verified | codec selftest |
| A | closed-form optimal per-row scale refit | implemented, zero extra bytes | codec selftest |
| A | post-hoc coding at fixed sub-bit rate | **SEALED NEGATIVE — at the RD floor** | probe, 15 real cells |
| B | M10 same-budget Doctor (sparse residual) | **implemented, measured on real weights** | `S64_doctor` |
| D | M05 + M09 structural expert reduction | **implemented, exact ledger, forward pending** | `S_STRUCTURAL_PLAN.json` |
| — | 1200-token disjoint routing calibration | in flight | replaces a contaminated 88-token sample |

## Lane A: the wall, measured

15 real cells (5 layers × 3 organs × 8 experts) at the exact geometry of the collapsed R2 arm.
`RD floor` = memoryless-Gaussian rate–distortion bound `sqrt(2^-2R)` at the identical index rate.

| cell | index bpw | baseline | scale-invariant | RD floor | row-norm span |
|---|---|---|---|---|---|
| L0 gate | 0.625 | 0.7214 | **0.6435** | 0.6484 | 15.54 dec |
| L0 up | 0.625 | 0.7260 | 0.6480 | 0.6484 | 15.47 dec |
| L23 gate | 0.625 | 0.7285 | 0.7028 | 0.6484 | 5.44 dec |
| L46 gate | 0.625 | 0.7240 | 0.7028 | 0.6484 | 2.11 dec |
| L70 gate | 0.625 | 0.7041 | 0.7029 | 0.6484 | 0.40 dec |
| L93 gate | 0.625 | 0.7156 | 0.7027 | 0.6484 | 10.44 dec |
| L0 down | 0.156 | 0.7460 | 0.7331 | 0.8974 | 1.67 dec |
| L46 down | 0.156 | 0.9266 | 0.9155 | 0.8974 | 0.88 dec |

Two findings, both against expectation:

1. **The sealed "94 % single-codeword" premise is false.** Measured share: 0.0267 mean,
   0.047 max, never close to 0.94. The declared `R5_rownorm_strat` lever had nothing to fix.
   Retired to the atlas as `row_norm_stratification_premise`.
2. **Post-hoc coding is at its information-theoretic bound.** L0 gate measures *below* the
   i.i.d. Gaussian floor; the rest are within 8 % of it. And the floor is catastrophic —
   `rel_error 0.65` on gate/up, `0.90` on down. Retired as `post_hoc_coding_of_frozen_weights`.

This closes Lane A **at these rates**. It is not, and may not be read as, a licence to bracket
upward. It is the quantitative reason the source must change under the same ceiling.

## Lane D: the inventory is a free variable

The ceiling constrains bits over the *original* weight count, so shrinking the expert inventory
and raising the survivor rate is budget-neutral. All arms verified by the law module:

| arm | complete BPW | exact | gate/up RD floor | down RD floor |
|---|---|---|---|---|
| S128 g1.25 d0.3125 | 0.951385 | — | 0.4204 | 0.8052 |
| **S64 g2.5 d0.625** | **0.948410** | `870957657/918334510` | **0.1768** | 0.6484 |
| **S64 + Doctor** | **0.999770** | — | 0.1768 | 0.6484 + repair |
| S32 g5.0 d1.25 | 0.947080 | — | 0.0312 | 0.4204 |

S64 is the first candidate in this campaign whose gate/up organ reaches a survivable
reconstruction regime at all. The A1 arm that collapsed 6/6 sat at floor 0.42; R2 at 0.65.

Declared costs of omission: survivor bitmap (`n_experts` bits per layer, a runtime table),
codebook amortized over **survivors only**, and the router's top-k restricted to survivors
(omitted logits masked to `-inf` *before* the top-k, then renormalized). That last item is a
genuine change to the model, which the law permits and requires declaring.

## Lane B: the Doctor rides the remainder

S64 leaves 0.0516 BPW under the ceiling. The Doctor spends all of it on the organ with the worst
remaining floor (`down_proj`), protecting the half of rows with the worst **relative residual
energy after the base pass** — the direct read on where the base representation lost the function.

Measured on real `down_proj` at layer 46: `rel_error 0.7038 → 0.6074`, +0.319 organ bpw.
Complete artifact: **0.999769787 BPW**, legal.

## What is NOT claimed

No capability. Every number above is weight-space or byte-space. Only a real parent-vs-packed
forward on the frozen holdout may select a frontier, and the parent reference logits are cached
and verified healthy for all six prompts.
