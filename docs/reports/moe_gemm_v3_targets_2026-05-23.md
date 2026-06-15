# MoE GEMM v3 targets — 2026-05-23

Read-only design pass over the M3 hot-spot kernels. No code changes.
Prioritises kernels by GPU-time share and what the existing v2t/gu_v2
optimisation toolbox can usefully apply.

## M3 hot-spot top 5 (decode 64-tok, contention-confounded but the
ratios are stable)

| % GPU | Kernel | Path | Block size | Status |
|------:|---|---|---|---|
| 26.6 | `moe_batched_gemm_q4_indexed_v2t_gu_v2` | gate+up routed | Q4_K 256-element | already gu_v2 — all 4 tricks applied |
| 12.3 | `mla_decode_kernel` | attention | f32 KV | Q8 KV port pending (separate plan) |
| 10.8 | `rmsnorm_gemv_f16w_attn_pinned_v2t` | rms+gemv | fp16 W / fp32 x | fused-add-rmsnorm exists at 1 site, 5 left (Session F) |
| 10.2 | `moe_batched_gemm_q5_0_indexed_v2t` | gate+up shared | Q5_0 32-element | only v2t, no gu_v2 |
| 9.5  | `gemv_f16_simdmat` | o_proj + others | fp16 W × fp32 x | already simdgroup-matrix MMA |
| 7.5  | `gemv_f16` | lm_head + small | fp16 W × fp32 x | LM head 4% of decode; bounded |
| 6.3  | `moe_batched_gemm_q8_0_indexed_v2t` | routed down | Q8_0 32-element | v2t + w2 sketch (env-gated) |

## Candidate v3 targets — sketches only

### T1. `moe_batched_gemm_q5_0_indexed_v2t_gu`  (NEW — biggest unmined lever)

**Why**: 10.15 % of GPU time and no gu_v2-class kernel exists for Q5_0.
The two structural tricks from Q4_K's gu_v2 transfer cleanly:
  - **Gate+Up fusion** in one kernel pass (currently the v2t kernel runs
    twice — once for gate, once for up — with two separate x_cache
    preloads and two separate dispatches). The shared-down phase Q5_0
    runs gate AND up over the SAME shape (2048×2816), so a fused kernel
    halves the x_cache preload bandwidth and the dispatch overhead.
  - **Block-pair processing** (2 Q5_0 blocks per inner iteration). Load
    2×22 bytes of weight data, hoist 2 fp16 d's + 2 qh uints into
    registers, do 2 MADs per lane. Halves outer-loop iteration count.

**Doesn't transfer**:
  - "Paired nibble" (Q4_K-specific — Q5_0's nibble layout is different).
  - "Sumy correction trick" — Q5_0 has no min-offset, so no correction
    term to amortize.
  - "Scale pre-load" — Q5_0 already loads d once per block.

**Estimated effort**: ~1 week design + implement + parity + bench gate.
**Estimated win**: gate+up fusion alone has saved 1.5–2 % e2e in Q4_K
historical data; Q5_0 share is 10.15 % so a 15 % kernel-internal speedup
maps to +1.5 % e2e ≈ +0.4 dec_tps. Block-pairing adds maybe +5 %
internal. **Total projection: +0.5–0.7 dec_tps**. Above the +1 tps gate
when stacked with T2 below.

**Risk**: Q5_0's qh layout (4 bytes representing 32 high-bits, one per
weight) means the simdgroup-divergent path `(simd_lane < 16u)` is the
critical-path hazard. A v3 should either: (a) read 2 packed bytes per
simdgroup so lanes 0–15 get one byte and lanes 16–31 get another (no
branch), or (b) precompute the lane→byte map in a constant table. Option
(a) is cleaner.

### T2. `moe_batched_gemm_q8_0_indexed_v2t_gu`  (NEW)

**Why**: 6.27 % GPU. Same gate+up fusion argument as T1. The fused
kernel preloads x_cache once for gate AND up, then runs two parallel
partial accumulators in registers (one per matrix).

**Doesn't transfer**: scale pre-load (already trivially 1 fp16 d per
block), paired nibble (Q8_0 has 1 byte per weight, not nibble-packed),
sumy correction.

**Estimated effort**: ~3–5 days (Q8_0 path is simpler than Q5_0).
**Estimated win**: 0.3–0.5 dec_tps (lower than T1 because Q8_0 share is
lower).

**Risk**: low — the inner loop is already tight, so the win is mostly
on dispatch and x_cache reuse. If dispatch overhead isn't the bottleneck
in the real trace, this sketch will benchmark flat. Microbench the
isolated "dispatch + x_cache fill" cost before committing.

### T3. Revive Session J `_v2t_w2` — extend to BOTH gate+up matrices

**Why**: `_v2t_w2` already exists and showed +1.33 % single shape but
was env-gated because it was only the **down** matrix (`y = silu(gate)
* up @ down`). If we apply the same 2-rows-per-simdgroup trick to a
**fused gate+up v3 kernel** (T2 above), we halve x_cache rebuild AND
halve the TG grid. Could compound.

**Estimated effort**: 1 week stacked on top of T2.
**Estimated win**: +0.2–0.4 dec_tps on top of T2's projected +0.3–0.5.

### Skip — already optimised

- **Q4_K gu_v2** (26.6 %) — already has all 4 tricks. Going further
  requires moving to simdgroup_matrix MMA at fp16 weight × fp32 x, which
  has been investigated (memory `v110_path30_findings.md`) and showed
  −14 % from register pressure. **Don't relitigate.**
- **MLA decode** — handled by Q8 KV port (separate plan).
- **rmsnorm_gemv_f16w_attn_pinned_v2t** — handled by Session F fusion
  (5 sites left, separate ticket).
- **gemv_f16_simdmat** — already at MMA peak for the shape.
- **gemv_f16** (LM head) — 4 % of decode; vocab-prune is what shrinks
  this, not kernel work.

## Suggested order if user gives the green light

1. **T2** first (Q8_0 gate+up fusion). Lower risk, faster to land,
   teaches us the gate+up fusion machinery before applying it to the
   structurally trickier Q5_0.
2. **T1** (Q5_0 v3 = gate+up + block-pair). Bigger projected win but
   structurally harder.
3. **T3** (w2 stacked on T2). Only if T2 benches positive and the v3
   kernel still has TG-grid headroom on M3 Pro.

**Total stacked projection: +1.0–1.6 dec_tps over ~3 weeks.**

Combined with the L1 +1.55 floor and a successful Q8 KV port (+1–5 if
the prior worktree's +1.6/+2.1/+2.5 % at 16/64/256 tok holds), this
gets us into the low 30s. Not 75. **path_to_75_v2.md's 6–10 week
projection still stands.**

## What this analysis is NOT

- It is not a microbench gate. None of these projections come from new
  numbers — they're back-of-envelope ratios from M3 + historical
  Session J/F deltas. Real numbers require parity-test + paired bench.
- It is not a commitment to land any of these. The user decides.
- It is not a substitute for the kernel sketch-→-parity-→-bench loop
  described in `feedback_kernel_parity_gate.md`. Each T# above needs
  its own parity test before any bench delta is trusted.
