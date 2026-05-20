# Phase E E.0.a — MLA kbatch K-sweep — NEGATIVE

**Date:** 2026-06-04
**Branch:** `claude/dreamy-golick-d54ff8`
**Commit anchor:** see phase E E.0.a commit
**Verdict:** **FAIL.** Both pre-validation gates red. Phase E (tree
decode) is killed here. Do not write E.0.b, E.1–E.5.

## Result in one paragraph

The MLA kbatch kernel does not scale flat with K. Per-dispatch cost
grows super-linearly across K=4 → K=8 on V2-Lite shape: K=8 is 2.34×
K=4 (gate required <1.5×), and linear extrapolation puts K=16 at 4.7×
K=4 (gate required <2.5×). Because tree decode amortization assumes
verifier cost-per-token is roughly flat as K grows, the underlying
physics on this kernel / hardware does not support the +30-50 dec_tps
Phase E lever. The original plan budgeted MEDIUM-LOW confidence with
this exact gate; the gate fired its protective function.

## Bench data

V2-Lite MLA shape: `n_heads=128, kv_lora_rank=512, qk_nope=128,
qk_rope=64, v_head=128, seq_len=256`. Clean-window M3 Pro 18 GB.
200 iterations × 50 warmup per K.

| K | mean μs | p50 μs | p99 μs | min μs | ratio K/K=4 |
|---|---------|--------|--------|--------|-------------|
| 1 | 1194.4  | 1088.7 | 1987.0 | 975.1  | 0.279       |
| 2 | 2085.4  | 1987.8 | 2828.8 | 1729.9 | 0.487       |
| 4 | 4284.8  | 4244.8 | 5249.4 | 3813.9 | 1.000       |
| 8 | 10024.9 | 10214.0 | 11841.7 | 8520.5 | **2.340**   |

Segment slopes (mean μs / additional K):
- K=1→K=2: 891.0 μs per K (sub-linear; weight + KV reuse working)
- K=2→K=4: 1099.7 μs per K (near-linear)
- K=4→K=8: 1435.0 μs per K (super-linear; getting worse)

The slope bends **up** between K=4 and K=8. This is the opposite
direction from what the lever requires.

Linear-fit extrapolation (least-squares through 4 points):
- best-fit `mean ≈ 1261.6 × K − 66.8` (μs)
- K=16 projected: **20119 μs** → K=16/K=4 = **4.70×**

Both extrapolation and segment-slope analysis agree: the kernel's
scaling is structural, not noise. Note that since the actual curve
bends up beyond K=4, the linear extrapolation is optimistic — the
real K=16 (if the kernel could run it, which it can't per the
`k_batch ∈ [1,8]` TG-memory cap) would likely be worse than 4.7×.

## Gate evaluation

| Gate | Threshold | Measured / Extrapolated | Result |
|------|-----------|-------------------------|--------|
| K=8/K=4 < 1.5  | direct measurement | 2.340  | **FAIL** |
| K=16/K=4 < 2.5 | linear extrapolation | 4.70 | **FAIL** |

Either gate red ⇒ phase killed per `phase_e_tree_decode_v2.md`.

## Why this happened — diagnostic

Reading `mla_decode_kernel_fc_kbatch.metal`, three K-fold loops
appear in the kernel body:

- Phase 0 — `q_nope_proj_k[kk]` computation: K-fold loop reading
  weights once and accumulating K query projections per row.
- Phase 1 — scores: K-fold loop with `seq_len` inner reads of c_kv.
- Phase 2/3 — softmax + final V projection: K-fold loops.

At K=4, weight-read amortization works (sub-linear K=1→K=2 slope).
At K=8, two threadgroup buffers `q_nope_proj_k` and `c_kv_wt_k` each
reach `8 × 512 × 4 = 16 KB` for a total of 32 KB — at the M3 Pro
per-core TG-memory ceiling. Once at the ceiling, the SM is forced to
serialize threadgroups (TG-occupancy drops), and the kernel pays a
~2.3× per-K cost penalty.

In other words: the K=1→K=4 regime is the kernel's sweet spot where
weight reuse pays back. K=8 is already past the architectural cliff.
K=16 is not even reachable on the existing kernel (would need a
tile-streaming rewrite). The L7 lesson restated: "the eliminated
intermediate-buffer traffic didn't matter; per-route-TG geometry
under-utilized the GPU" — same shape of failure, different kernel.

## What would unblock Phase E

A successful Phase E would need a different kernel architecture that
keeps TG memory roughly constant in K. Options that might restore
flat scaling:

1. **Streaming q_nope_proj_k.** Instead of caching all K projections
   in TG memory, stream a tile (e.g. K_tile=2) at a time. Recomputes
   weight reads K/K_tile times but stays under the TG ceiling. Risk:
   recompute cost ≈ weight-reuse savings; may not net positive.
2. **Online (flash-style) softmax.** Eliminates the `scores_scratch`
   device round-trip and reduces TG pressure. The kernel comment
   (line 17–19) explicitly notes this as a "future optimization."
   Probably necessary for K>4 to be viable.
3. **Different K layout.** Per-K threadgroup instead of per-head
   threadgroup, with a final reduction. Trades occupancy structure;
   may or may not help V2-Lite's 128-heads pattern.

Each of these is a multi-day kernel rewrite. They turn Phase E from
"chase a +30-50 dec_tps lever" into "rewrite the MLA kernel to
support tree decode at all" — a much bigger commitment with no
guaranteed payoff. The whole point of E.0.a was to surface this
choice cheaply, and it did.

## Next-action recommendation

Per the v2 plan's decision flowchart:
- **STOP Phase E.** Do not write E.0.b (chunked-K MoE union kernel)
  or E.1–E.5. The MoE side cannot save a phase whose MLA side has
  already failed.
- **Pivot to Phase F (medusa)** as the highest-value remaining lever
  on the path-to-150 board. F is structurally independent of the
  per-K verifier scaling that just failed (medusa heads emit
  multiple tokens per forward pass; they do not require K-batched
  verifier kernels).
- **Or revisit L8 training** if Phase F's clean-overnight capture
  window is not available. L8 iter 4 halted at 0% K=2 chain accept;
  a new iter that reaches ≥20% chain accept would re-open Phase L5.

## Artifacts

- New bench fixture: `crates/dismantle-core/src/kernel_bench.rs`
  function `bench_mla_kbatch` (~110 LoC).
- ALL_KERNEL_NAMES entry: `"mla_decode_kernel_fc_kbatch"`.
- Env-var overrides: `DISMANTLE_MLA_BENCH_k`,
  `DISMANTLE_MLA_BENCH_seq_len`.
- Bench is reusable for any future MLA kernel variant — drop in a
  new kernel name and re-run the K-sweep to validate against the
  same gate.

## What worked

The pre-validation gate worked exactly as designed. Half a day of
fixture work surfaced a kernel-scaling cliff that would otherwise
have cost ~1-2 weeks of E.1–E.5 implementation before showing up
in the final bench. The L7 lesson — "parity-only validation is
insufficient" — was generalized in v2 to "every milestone ships
its own bench fixture, and pre-validation gates kill the phase
before integration." The E.0.a result is the first proof point
that this discipline pays.
