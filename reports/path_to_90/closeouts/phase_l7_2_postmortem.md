# Phase L7.2 post-mortem — fused vs chained, 3.8× regression root cause

**Status:** NEGATIVE confirmed. `moe_expert_pair_fused` ships parity-green
but benches 3.69-3.77× slower than the chained union pipeline at every
N. This doc identifies the root cause and codifies forward rules so
follow-on kernel work doesn't repeat the failure mode.

**Inputs:**
- Bench commit `332979b` — matched-pair MoE bench fixtures, 200 iter, V2-Lite shape (2048 / 1408 / 2048).
- Kernel commit `8073a9e` — `moe_expert_pair_fused.metal`, dispatcher, parity.
- Chained reference: [moe_gate_up_union_v2t](crates/dismantle-core/src/kernels/parallel_k.rs:747) + `moe_down_union_v2t` (the existing union pipeline).

## What the bench actually showed

```
shape 2048x1408 (V2-Lite gate_up + down), 200 iter, Claude alive:

  N=6   chained: 1019.8 µs  /  fused: 3847.5 µs   (3.77× slower)
  N=24  chained: 1045.3 µs  /  fused: 3851.6 µs   (3.69× slower)
```

Two facts the numbers force:

1. **Neither path is compute-bound.** Going from N=6 → N=24 (4× more routes) costs the chained kernel +2.5% and the fused kernel +0.1%. Both are essentially flat in N. Real compute scaling would be linear.

2. **The regression is structural, not data-dependent.** Same 3.7× gap at every N, ergo the cause is in per-launch fixed cost, not in any work that scales with route count.

## Why the original "expert weight re-read" hypothesis was wrong

The L7.2 design doc ([moe_expert_pair_fused.metal:18-26](crates/dismantle-core/shaders/moe_expert_pair_fused.metal)) anticipated a trade-off: lose per-expert weight reuse (~56 MB extra DRAM traffic at K=4, ~60% overlap), gain ~135 KB of `routed_act` writeback savings. If that trade had been the dominant factor, the regression would scale with N (more routes → more redundant reads). It doesn't. So the trade-off framing missed the actual mechanism.

## Root cause: chronic GPU under-occupancy

**Chained grid** ([parallel_k.rs:795-799](crates/dismantle-core/src/kernels/parallel_k.rs:795)):
```
Grid: (ceil(rows/8) × 256, n_experts, 1)   →  (176 × 256, 64, 1)
Total TGs: 176 × 64 = 11,264
```

Early-returns prune the inactive experts (those outside the union), but the dispatch still floods the machine with thousands of threadgroups. The hardware sees a fully saturated pipe.

**Fused grid** ([moe_expert_pair_fused.metal:28-29](crates/dismantle-core/shaders/moe_expert_pair_fused.metal:28)):
```
Grid: (1, n_routes, 1)                     →  (1, 6, 1)  and  (1, 24, 1)
Total TGs: n_routes
```

At N=6 the fused kernel launches 6 TGs on an 18-core M3 Pro. Each core wants ~4 concurrent TGs to hide memory latency. The target is ~72 in-flight TGs. The fused kernel fills ~8% of that. At N=24 it's still under 35%.

Inside each fused TG, 8 simdgroups serialize 176 row-batches in Stage A (`ceil(1408/8) = 176` outer iters) and 256 in Stage B (`ceil(2048/8) = 256`). That's a long sequential dependency chain on each of a small handful of TGs. The chained kernel breaks the same work across the row dimension into the *grid*, not into a *loop inside one TG*.

The intermediate-write savings (~135 KB/token) sit in a memory budget that's already cheap on a saturated machine. The savings can't pay back the occupancy loss.

## Three patterns L7.2 threw away

L7.2 collapsed the row dimension into a single TG per route in order to share `act_cache` across Stage A and Stage B. Doing so abandoned three patterns that already had bench evidence in this repo:

1. **8-rows-per-TG geometry** — proven in `gemm_q4_k_m_v3_8r` (Phase B), `v3_xtg` (L7.1), and `v3_xtg_sumy` (Stage 0.5, [c0fc428]). Standalone GEMV's best-known geometry.
2. **n_experts y-grid + segment-scan iteration** — the chained union kernel's parallelism source. Lets the dispatcher fill the GPU even at small route counts.
3. **Per-stage kernels, not per-stage fusion** — two simpler kernels with high parallelism beat one fused kernel with low parallelism at this shape and machine.

## Codified rules for follow-on kernel work

These go into [methodology_distilled_post_f2.md](reports/path_to_90/plans/methodology_distilled_post_f2.md) as patterns 21-24 on the next attended pass.

### R1 — Occupancy floor

A new kernel design is rejected at architecture time if its dispatch grid produces fewer than **72 concurrent TGs** at the smallest expected N on M3 Pro (18 cores × ~4 TGs/core target). Compute this from the grid math before writing the shader. The floor is defensible, not measured — tighten with a Metal occupancy trace once such a trace is in the toolkit.

### R2 — Default to 8-rows-per-TG geometry

Use the proven `v3_8r` / `v3_xtg` / `v3_xtg_sumy` row-batched layout unless the design has a quantified trade explaining the deviation. L7.2's deviation was "share threadgroup memory across stages" and the trade lost. Future deviations need a quantified prediction *and* a kill-switch geometry to fall back to.

### R3 — Bench gates wiring, not the other way around

L7.2 landed the dispatcher + parity in `8073a9e` and the bench result in `332979b` 36 minutes later. The dispatcher edit shipped before the regression was visible. The active profile didn't switch (`expert_pair_fused` is reserved-but-unused), so no harm done in this case, but the ordering is unsound. **Rule:** bench commit before wiring commit, with the bench result quoted in the wiring commit message.

### R4 — Fusion isn't free

Fusing kernels reduces global writeback but tends to reduce parallelism: shared state forces fewer TGs (or wider TGs, or longer per-TG dependency chains). Only fuse when the post-fusion design still clears R1 *and* preserves R2's row geometry. If neither holds, the fusion's bandwidth savings will not pay back the dispatch loss at any realistic N.

## What stays on disk; what gets a kill-switch

- **Kernel + parity test stay.** They're parity-gated, dormant (not in any active schedule string), and serve as a cautionary fixture for future fusion attempts. Deleting them costs more than the ~440 LoC they take.
- **No profile change needed.** `expert_pair_fused` is reserved-but-unused in the active profile, so nothing is shipping the slow path to users.
- **Bench fixtures stay.** The matched-pair bench in `kernel_bench.rs` is the cheapest way to re-evaluate if a future occupancy fix (e.g. persistent-thread variant) attempts the same fusion shape.

## What this means for the remaining L7 plan

[phase_l7_kernel_rewrites.md](reports/path_to_90/plans/phase_l7_kernel_rewrites.md) listed three kernels: `moe_expert_pair_fused` (done, dead), `gemv_q4_k_v3_mlx` (L7.E, MLX-ref-blocked), `moe_batched_gemm_q4_indexed_v3` (L7.D, MLX-ref-blocked). With L7.2 dead and L7.D / L7.E both gated on the MLX-LM reference drop, the L7 plan is **stalled until the reference lands in `reports/path_to_90/mlx_lm_ref/`**. See [phase_l7d_plan.md](reports/path_to_90/plans/phase_l7d_plan.md) for the architected pickup once that drop happens.

## Followups (next attended session)

1. Move R1-R4 into `methodology_distilled_post_f2.md` as patterns 21-24 with cross-refs to this doc.
2. Get the MLX-LM reference (`mlx_lm/models/quantized_linear.py` + relevant Metal kernels from MLX itself) dropped into the worktree so L7.D / L7.E can leave architecture and enter implementation.
3. Audit other prior fused-kernel attempts in the repo (v2.3.0 megakernel notes per [v230_icb_dead.md] memory) for occupancy patterns relevant to a future Stage-B-only fusion.
4. Optional: add a debug counter to the chained union dispatcher that logs `n_active_experts` per token. The bench at N=24 is plausibly already at the union's compute floor; knowing the *actual* active expert count distribution in prod traces would let R1's floor be tightened.
