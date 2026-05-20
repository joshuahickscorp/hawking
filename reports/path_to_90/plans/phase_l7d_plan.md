# Phase L7.D plan — `moe_batched_gemm_q4_indexed_v3`

**Status:** ARCHITECTED, IMPLEMENTATION BLOCKED on MLX-LM reference drop.
**Predecessor lesson:** [phase_l7_2_postmortem.md](reports/path_to_90/closeouts/phase_l7_2_postmortem.md) — apply R1-R4 throughout.
**Original spec:** kernel #3 in [phase_l7_kernel_rewrites.md](reports/path_to_90/plans/phase_l7_kernel_rewrites.md) (~180 lines).
**Parent target:** path-to-100. L7.2 dead; L7.D + L7.E are the remaining L7 levers.

## What L7.D is supposed to do

The parent plan calls L7.D "MoE Q4_K_M GEMV with MLX patterns" — a rewrite of [moe_gate_up_union_v2t](crates/dismantle-core/shaders/moe_union_expert.metal:47) that ports MLX-LM's Q4_K_M kernel inner-loop pattern (simd-shuffle paired-nibble + whatever else MLX does that the v3_xtg_sumy family doesn't). The outer structure — union dispatch, 8-rows-per-TG, n_experts y-grid, segment-scan iteration over routes — is retained because the post-mortem proves it's the right structure on this hardware (R1, R2).

L7.D is therefore an **inner-block rewrite**, not a restructuring.

## What blocks implementation right now

The MLX-LM reference is not in the worktree. The parent plan calls out the drop location:

> (2) requires external code reference (MLX-LM's `mlx_lm/models/
> quantized_linear.py`). User needs to drop a copy into
> `reports/path_to_90/mlx_lm_ref/` before this work.
> — [l7_tomorrow_pickup.md:150-152](reports/path_to_90/plans/l7_tomorrow_pickup.md:150)

Without seeing the MLX kernel we can't architect the inner-block pattern. Anything we write would be guessing at what "MLX-style" means, and L7.2 already paid the cost of guessing-then-shipping.

**Two paths forward:**

- **Preferred** — drop the MLX reference, then this plan exits architecture and enters implementation.
- **Alternate** — pick a different L7 lever that doesn't depend on the reference. See "What we could do *instead* without MLX ref" below.

## Architecture (the parts we can fix now)

Apply R1-R4 from the post-mortem so the inner-block work doesn't accidentally regress occupancy:

### Geometry (R1, R2)

Mirror `moe_gate_up_union_v2t` exactly:
```
Grid:  (ceil(rows/8) × 256, n_experts, 1)   // rows = routed_mid = 1408 at V2-Lite
TG:    (256, 1, 1) = 8 simdgroups × 32 lanes
shmem: K × cols × 4 bytes                    // x_cache_all
```

At V2-Lite this dispatches `176 × 64 = 11264` TGs at any N, comfortably clearing R1's 72-TG floor by >150×. Early-return prunes inactive experts via `segment_starts[expert] == N`. We get the union pipeline's parallelism for free.

### What changes vs the existing `moe_gate_up_union_v2t`

Only the per-block decode + MAD inner sequence. The framing (cooperative x_cache load, segment iteration, sumy correction, 8-rows-per-TG simdgroup assignment) is preserved. Concretely, the diff target is lines 47-end of `moe_union_expert.metal` — the body of `moe_gate_up_union_v2t`, the inner block loop.

What we expect to swap in (subject to confirmation against the MLX ref):

- **simd-shuffle-based nibble distribution** — MLX-LM reportedly broadcasts the packed nibble byte across the simdgroup with `simd_shuffle` and lets each lane mask its own pair, instead of each lane reading its own byte directly from `w_q4[bo + 16 + pi*32 + lane]`. The motivation is to coalesce the 32-byte read into one transaction instead of 32 scattered loads. Whether this actually wins on Apple Silicon (where L1 already coalesces sequential lane reads cheaply) is the empirical question the MLX ref answers.
- **Per-lane decode locality** — possibly a different mapping from `(b, pi, lane) → element_index` that improves MAD pipelining.

These are hypotheses, not designs. The MLX kernel is the source of truth.

### Parity test (R3, R4)

`crates/dismantle-core/tests/moe_batched_gemm_q4_indexed_v3_parity.rs` (~150 LoC).

- Synthetic Q4_K_M weights via `synthetic_q4_k_bytes` (reuse from `path_b_parity.rs`).
- Synthetic union routing tables — 64 experts, K=4, top_k=6, with controlled overlap. Two cases:
  - **Sparse**: N=6 active routes, each on a unique expert (no overlap, smallest realistic case).
  - **Dense**: N=24 active routes spanning ~10 experts with ~2-3 routes/expert (overlap that exercises the segment-scan inner loop).
- Reference: existing `moe_gate_up_union_v2t` output.
- Candidate: new `moe_batched_gemm_q4_indexed_v3` output.
- Tolerance: `atol=1e-3` fp16. Both bit-identical and within-tolerance regimes acceptable; tolerance is the gate.

### Bench gate (R3)

Land bench commit BEFORE dispatcher wiring commit. Bench fixtures reuse the matched-pair pattern shipped in `332979b`:

- `moe_gate_up_union_v2t` (existing) vs `moe_batched_gemm_q4_indexed_v3` (new), same N sweep (6, 24, 48).
- Acceptance: ≥10% mean reduction at N=6 AND ≥5% at N=24, both with overlapping noise bars from 200-iter mean ± stdev. Anything weaker is noise-floor; do not wire.

### Wiring (gated on bench pass)

If acceptance hits:
1. Add `"q4_k_v3"` (or whatever schedule string maps to MLX-style) to the `routed_gate_up_schedule` enum in [profile.rs](crates/dismantle-core/src/profile.rs).
2. Add the dispatcher branch in [kernels/mod.rs](crates/dismantle-core/src/kernels/mod.rs) mirroring `moe_gate_up_union_v2t`'s call site.
3. Flip the active profile (`profiles/deepseek-v2-lite-q4.m3pro18.json`) once a clean-window e2e bench confirms the per-shape win translates to dec_tps.
4. Bump shader_hash in the same wiring commit.

The wiring is small (~30 lines kernels/mod.rs + 2 lines profile.rs + 1 line profile.json) and follows the existing schedule-string pattern. No structural risk.

## What we could do *instead* without the MLX ref

If the MLX drop is going to take more than one attended session, here are the L7-adjacent levers that don't depend on it:

1. **L7.D-alt: persistent-thread Stage-B fusion (speculative).** Re-attempt the L7.2 fusion goal (eliminate the `routed_act` round-trip) but with a persistent-thread design: dispatch exactly 72 TGs (matching the M3 Pro occupancy floor), have each TG cooperatively chew through routes from a shared atomic counter. Each route is processed Stage A → barrier → Stage B in the TG. This decouples TG count from route count, satisfying R1 at any N.
   - **Risk:** memory references hint v2.3.0 megakernel attempts were ruled out for related reasons. Worth re-reading [v230_icb_dead.md] before pursuing.
   - **Cost:** ~250 LoC new shader + matched-pair bench gate.
   - **Confidence:** LOW. This is the lever L7.2 *should* have been if R1 had been a rule then.

2. **L7-orthogonal: argument-buffer rollup for routed dispatch.** Per-kernel argument-buffer building shows up in profile traces. If the union dispatch's per-token argbuf cost is measurable (>5% of routed step time), a one-shot argbuf reused across all routed dispatches could trim it. Pure plumbing, no kernel work.
   - **Confidence:** MEDIUM-LOW. Needs a profiler trace to confirm the argbuf cost is real before committing.

3. **L5 chain-pipeline restructure** (per [phase_l5_chain_pipeline.md](reports/path_to_90/plans/phase_l5_chain_pipeline.md)). Out of L7 scope but the path-to-100 retool sequenced it after L7 anyway. If L7 is genuinely blocked, jumping forward in the sequence is cheaper than re-architecting L7 around the block.

Recommendation: **prefer (3) over (1) or (2).** L7.D-alt is high-risk re-treading dead ground; argbuf rollup needs profiler evidence; L5 has its own plan doc and is the queued lever. Skipping forward in the sequence is fine as long as the MLX ref drop is tracked in followups.

## Pickup checklist (when MLX ref lands)

When `reports/path_to_90/mlx_lm_ref/quantized_linear.py` (and any companion Metal sources) appear in tree:

1. Read the MLX kernel inner block. Identify the specific divergence from `gemm_q4_k_m_v3_xtg_sumy` (which is the closest extant baseline).
2. Update this doc's "What changes vs the existing" section with the actual MLX pattern, not the hypothesis.
3. Implement `crates/dismantle-core/shaders/moe_batched_gemm_q4_indexed_v3.metal` (~180 LoC).
4. Implement parity test (~150 LoC).
5. Implement matched-pair bench fixture (extend `kernel_bench.rs` following the L7.2 bench pattern in `332979b`).
6. Single commit: shader + parity (no wiring, no bench).
7. Run bench. Land bench commit (with numbers in commit message).
8. If pass: land wiring commit (with bench numbers quoted).
9. If fail: write `phase_l7d_blocked.md`, halt, hand back to attended session.

Per-step ceiling: 60 min. Total budget: 4 hr per dismantle CLAUDE.md. If step 1's diff is more invasive than "inner-block swap" (e.g. MLX uses a fundamentally different threadgroup layout), STOP and re-architect — that's a different kernel, not L7.D.

## What L7.D is *not*

- Not a restructuring of the union pipeline (that's L7.2-style and we proved it's dead).
- Not a "while we're here" sweep of other MoE kernels — only `moe_gate_up_union_v2t` is in scope. `moe_down_union_v2t` stays as-is unless a separate L7.D-down plan is written.
- Not a profile flip in the same commit as the shader. R3 enforces bench-then-wire.
