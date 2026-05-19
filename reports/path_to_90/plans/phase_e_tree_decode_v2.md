# Phase E v2 — tree decode (post-L7 revision)

**Supersedes:** `phase_e_tree_decode.md`. The original plan is preserved
but its confidence assumptions need adjustment in light of what L7
taught us this session.

**Goal:** +30-50 dec_tps (1.5-1.8× chain speculation multiplier).
**Engineering est:** 1-2 weeks focused work, +1-2 days for
pre-validation.
**Confidence:** **LOW** until pre-validation passes. Originally rated
MEDIUM-LOW; downgraded based on the L7 track record.

## Why this plan was revised

In session 2026-05-19 / 2026-06-03 the L7 phase shipped two kernel
rewrites that both passed parity on the first try and both lost the
bench by margins ranging from 5% (sumy on LM head) to 380% (fused MoE
on V2-Lite expert shape). The original L7 plan budgeted "MEDIUM"
confidence for the GEMV variant and "MEDIUM" for the fusion; both
turned out wrong, in opposite directions:
- The sumy variant: the dm-MAD savings the plan counted on didn't
  materialize; the extra simd_sum synchronizations dominated.
- The fused MoE: the eliminated intermediate-buffer traffic the plan
  counted on didn't matter; the per-route-TG geometry under-utilized
  the GPU by 4×.

Phase E is a much bigger architectural change than either L7 lever.
The original plan's "this is empirical" hedge is the right framing;
this v2 plan adds explicit gates and a pre-validation experiment.

## Pre-validation experiment (NEW — do this first)

Before writing any of E.1-E.5, run a synthetic K*B verifier-cost
bench. The question this answers: **does the verifier's per-token
cost stay roughly flat as K grows from K to K*B?** If yes, tree decode
can amortize and the +30-50 dec_tps lever is reachable. If
verifier-cost-per-token grows linearly with K (i.e., K*B is ~B× slower
per token than K), tree decode adds no leverage and the phase should
not ship.

### Hard constraint discovered while writing this plan

The existing `moe_gate_up_union_v2t` kernel caches all K activation
vectors in threadgroup memory: `x_cache_all = K × cols × 4 bytes`.
At V2-Lite cols=2048, K=4 already saturates the 32 KB TG-memory
budget. **The current union kernel CANNOT run at K > 4 on V2-Lite.**
Reaching K*B = 8 (B=2, K=4) requires either (a) a new chunked-K MoE
kernel that streams x in tiles instead of caching all K, or (b) a
tree-aware MoE kernel that exploits sibling x reuse.

This means the E.0 pre-validation experiment cannot extend the
existing union kernel as drop-in. Two ways to structure the E.0
fixture:

**E.0.a (cheaper, partial signal)** — bench only the MLA kbatch
kernel at K=4/8/16. MLA does not have the same TG-memory ceiling
(K appears as a separate query-batching dimension, not as a TG cache).
This answers half the question: does the ATTENTION verifier cost
scale flat with K? If MLA scales well, the next question is whether a
chunked-K MoE kernel can match. ~150 LoC, ~half a day.

**E.0.b (full signal)** — write a chunked-K MoE union kernel first,
then bench. This is ~400-500 LoC of new kernel work (which is itself
~1/3 of E.3's tree-batched MoE) plus the bench fixture. ~3 days. Most
of the work either advances E.3 or proves it impossible.

Recommended path: **E.0.a first.** If MLA scales acceptably (likely
based on the existing kbatch design), graduate to E.0.b. If MLA
doesn't scale well, the phase is dead at the cheapest gate.

### Original E.0 gates (kept for reference)

Concrete fixture (~250 LoC, ~1 day — assumes E.0.a only):
- Extend `mla_decode_kernel_fc_kbatch` bench fixture from K=4 to
  K∈{4, 8, 16}
- Bench at V2-Lite shape, K-sweep, contended-window (acceptable for
  ratio data per L7.2's experience)

Gate to proceed:
- K=8 must be < 1.5× slower than K=4
- K=16 must be < 2.5× slower than K=4

If either gate fails: tree decode probably doesn't win on V2-Lite.
Write the negative-result doc and pivot to Phase F or revisit L8
training. **Do not write E.1-E.5 without this gate green.**

If gates pass with MLA only: proceed to E.0.b (chunked-K MoE) before
E.1. The risk is that MoE-side scaling looks worse than MLA-side and
kills the phase later — better to surface that early.

If gates pass with both MLA and MoE: proceed to E.1 with confidence
that the verifier cost model supports the lever.

## What's already shipped (unchanged)

- Parallel-K verify (Branch 1) at K=4
- Branch 2 step 4 (parallel-k-union) — MoE expert union amortization

## Acceptance criteria (per-milestone, NEW)

Each milestone of E.1-E.5 ships **only if** a matched-pair bench
fixture shows the change is no worse than 1.0× the chain-decode
baseline at B=1 (parity-equivalent regime) AND the
forward-projection to B=2 looks positive. This means every milestone
adds its own bench fixture before the integration code is reviewed,
not after.

Acceptance criteria for the phase as a whole (from v1, unchanged):
- B=1, K=4 tree decode bit-identical to chain decode at K=4
- B=2, K=4 chain accept ≥ 60% (vs chain's projected ~45-55%)
- Clean bench: ≥1.3× dec_tps over chain decode at matched K
- All existing parity gates still pass

## Milestone breakdown (unchanged from v1, summarized)

- **E.0 — pre-validation bench** (NEW, ~1 day): the K-sweep above.
- E.1 — tree-shape draft head (3-4 days), Eagle4 + Python training.
  Gated on L8 having a working chain head — currently 0% at iter 4.
- E.2 — tree-batched MLA kernel (3-4 days).
- E.3 — tree-batched MoE kernel (2-3 days).
- E.4 — tree-decode verify loop in Rust (2 days).
- E.5 — parity + clean-window bench (1-2 days).

## Risks + mitigations (revised)

1. **L7 lesson: parity-only validation is insufficient.**
   *Mitigation:* every milestone ships its own bench fixture; no
   "ship the kernel + leave bench to the next session" pattern. The
   matched-pair bench fixture pattern (this session's
   `moe_expert_pair_chained` vs `moe_expert_pair_fused`) is the
   correct template.

2. **Tree overhead exceeds branch gain on V2-Lite MoE shape.**
   *Mitigation:* E.0 pre-validation gate. If verifier cost scales
   linearly with K, tree decode loses by construction.

3. **L8 chain accept blocker.**
   *Mitigation:* E.1 (head training) is gated on a working chain
   head. L8 iter 4 K=2 vector finished at 0% chain accept. Until a
   subsequent L8 iteration produces ≥20% K=4 chain accept, E.1
   should not start (the head will train on the wrong objective).

4. **Implementation complexity introduces parity bugs.**
   *Mitigation:* B=1 parity gate (must equal chain decode). Same as
   v1.

5. **Memory pressure** — tree drafts cost K*B activation buffers.
   *Mitigation:* same as v1 — verify within decode arena budget.

## Acceleration patterns applied (NEW emphasis)

- Pattern 1 (mid-flight signal): the pre-validation bench is the
  largest mid-flight signal available — it tells you whether the
  phase can succeed before you spend a week on it.
- Pattern 2 (smoke ≠ eval ≠ bench): explicitly NOT relying on parity
  as the only gate. Each milestone has its own bench fixture.
- Pattern 5 (architectural fix before hyperparameter sweep):
  pre-validation IS the architectural question; only proceed to
  hyperparameter tuning (B=2 vs B=4, training settings) after E.0
  proves the architecture works on V2-Lite hardware.
- Pattern 9 (code-vs-compute accounting): same 1410 LoC budget as
  v1, plus 250 LoC for E.0. Use the budget as a forcing function —
  if E.0 takes more than 1.5 days, scope it down.

## Decision flowchart

```
Start
  │
  ▼
E.0 pre-validation bench
  │
  ├── K=8 < 1.5× K=4 AND K=16 < 2.5× K=4?
  │        │
  │        Yes → proceed to E.1 (or E.2/E.3 if L8 still HALTed)
  │        │
  │        No  → STOP. Write negative-result doc. Pivot to Phase F
  │              or revisit L8 training.
  │
  ▼
E.1-E.5 in sequence, each with its own bench-fixture gate
  │
  ▼
E.5 GO/NO-GO with full B=2 tree decode dec_tps bench
```

## Next-session quickstart

```
1. Read this + acceleration_patterns.md + L7 closeout (the bench-fixture
   pattern is the template you want).
2. Write the E.0 pre-validation bench fixture (extend the existing
   K=4 fixtures to K=8 / K=16). ~250 LoC, ~1 day.
3. Run it under contention (Claude alive is fine for ratios).
4. If gates pass: proceed to E.1 only if L8 has a working chain head,
   otherwise jump to E.2 (kernel scaffolding).
5. If gates fail: write `reports/path_to_90/closeouts/phase_e0_negative.md`
   and pivot.
```
