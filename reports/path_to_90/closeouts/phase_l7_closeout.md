# Phase L7 closeout — kernel rewrites (Stage 0.5 + L7.2)

**Status:** SHIPPED kernels + parity gates. Stage 0.5 micro-benched —
**regression vs v3_xtg, do not flip**. L7.2 fusion parity-verified;
end-to-end bench requires live wiring (deferred).

**Branch:** `claude/dreamy-golick-d54ff8`
**Commits:**
- `c0fc428` — `path-to-150 L7 / Stage 0.5 — gemm_q4_k_m_v3_xtg_sumy kernel`
- `8073a9e` — `path-to-150 L7.2 — mixed-quant moe_expert_pair_fused kernel + parity`
- `e5c5435` — `path-to-150 L7 closeout — Stage 0.5 + L7.2 shipped, bench queued`
- (this commit) — `path-to-150 L7 bench — Stage 0.5 contended-window result, negative`

## What shipped

### Stage 0.5 — `gemm_q4_k_m_v3_xtg_sumy` (c0fc428)

New standalone Q4_K_M GEMV variant. Same `v3_xtg` geometry (8 simdgroups
× 1 row/simdgroup, cooperative threadgroup `x_cache`) plus the min-
correction `sumy` trick already in `v3_llama` and `moe_gate_up_union_v2t`:
precompute `simd_sum(xl[k])` per sub-block, accumulate `dm[k] * sumy[k]`
outside the nibble loop. Replaces 256 `dm * xl` MADs per row per block
with 8 `simd_sum` invocations + 8 MADs.

- Shader: `crates/dismantle-core/shaders/quant.metal:634` (~100 LoC)
- Dispatcher + public wrapper: `crates/dismantle-core/src/kernels/mod.rs`
- Opt-in via `gemm_q4_k_schedule = "v3_xtg_sumy"`; active profile still
  selects `v2t_gu_v2`, so the new kernel is dormant.
- Parity: `tests/q4_k_v3_xtg_sumy_parity.rs` — 4 cases at 1e-5 rel
  (basic 16×256 + 64×512, V2-Lite expert 10944×2048, LM head 4096×2048,
  cross-check vs `v3_xtg` at V2-Lite expert shape). All pass.
- shader_hash: `65e7588d…` → `b935ca3d…`, bumped in profile in same commit.

### L7.2 — `moe_expert_pair_fused` (8073a9e)

New mixed-quant fused MoE expert kernel. One TG per route; threadgroup
SRAM holds `x_cache` (hidden_in floats) and `act_cache` (routed_mid
floats). Stage A produces `act_cache = silu(W_gate_Q4K @ x) * (W_up_Q4K
@ x)` using paired-nibble decode + sumy correction; threadgroup barrier;
Stage B produces `y = W_down_Q8 @ act_cache`. Eliminates the
`routed_act` and `routed_out` global-memory round-trips between the
existing union-pipeline gate_up and down kernels.

- Shader: `crates/dismantle-core/shaders/moe_expert_pair_fused.metal` (236 LoC)
- Wired into `all_shader_sources()` after `SHADER_MOE_UNION_EXPERT`
- Parity-test-only dispatcher
  `dispatch_moe_expert_pair_fused_pinned` in `crates/dismantle-core/src/kernels/mod.rs`
- Parity: `tests/moe_expert_pair_fused_parity.rs` — 2 cases at 1e-3 rel
  (small 256/64/64, V2-Lite 2048/1408/2048). Both pass on first run.
- shader_hash: `b935ca3d…` → `1d71174f…`, bumped in profile in same commit.

## What deferred (and why)

1. **Live wiring of `moe_expert_pair_fused`.** The kernel is parity-gated
   and reachable via the new dispatcher, but it is not yet plumbed into
   the routed MoE dispatch path in `deepseek_v2.rs`. Wiring requires
   integration with the `topk_gate` → route-ids → segment-scan flow that
   the existing union pipeline owns. The schedule string
   `"expert_pair_fused"` is reserved; the live branch should be added
   only after a clean-window bench confirms wins at one of K=1 (greedy
   verify) or K=4 (chain).
2. **L7.4 broader MoE Q4_K_M v3 (`moe_batched_gemm_q4_indexed_v3`).**
   Original plan called for a third kernel that applies the x_cache +
   sumy + MLX patterns to the existing `v2t_gu_v2_fc` MoE batched GEMV.
   Deferred because (a) it is the lowest-confidence kernel in the plan
   (its win/loss depends on patterns we have not yet validated on
   real-model traces), and (b) a clean bench of the two shipped kernels
   should inform whether the broader rewrite is worth the engineering.
3. **Bench window (`tools/bench/path_to_125_bench.sh`).** Deferred to a
   follow-on session. Per-shape A/B with the new schedules enabled via
   `gemm_q4_k_schedule_per_shape` is the path; current acceptance gate
   is ≥5% wall improvement.

## Surprises / corrections to the original plan

1. **Routed down is Q8_0, not Q4_K_M, in the active V2-Lite profile.**
   The L7 plan doc assumed Q4_K_M end-to-end for the fusion kernel.
   The active `deepseek-v2-lite-q4.m3pro18` profile uses
   `routed_down_schedule = "v2t"` against a Q8_0 down tensor (confirmed
   via `routed_down_q8` in the existing `moe_block_batched_indexed`
   dispatcher at `crates/dismantle-core/src/kernels/mod.rs:1380`). The
   L7.2 kernel was redesigned to take Q4_K gate + Q4_K up + Q8_0 down
   in a single kernel — slightly more complex than the all-Q4_K design
   in the original plan but matches the actual model layout. Q8_0's
   32-element block aligns exactly with the simdgroup width, so the
   down stage's inner loop is cleaner than the Q4_K_M down would have
   been (cols=1408 doesn't divide the Q4_K_M 256-element block anyway).
2. **MLX-LM source reference not present.** The plan's Stage 0.5 section
   suggested copying MLX-LM kernels into
   `reports/path_to_90/mlx_lm_ref/` for in-repo reference. That dir
   doesn't exist. On closer reading the MLX-LM pattern (N_R0=1 +
   paired-nibble + sumy) maps cleanly onto `v3_xtg + sumy`, which is
   what shipped. The sumy reference came from in-repo
   `gemm_q4_k_m_v3_llama` and `moe_gate_up_union_v2t`, both of which
   already implement the trick.
3. **L8 monitoring session completed mid-haul.** Background pids 43980
   (training) and 52414 (autoiter) exited during this session.
   `tools/l8_autoiter.sh status` shows `iter4_k2_vector: 0.0% accept →
   HALT` (regression from step-400 mid-flight's 33.3% — the autoiter
   session's problem). With L8 done, the clean-window bench condition
   for the L7 deliverables is satisfied for whichever session runs
   them next.

## Net dec_tps delta

### Stage 0.5 (v3_xtg+sumy) — bench result: **regression, do not ship**

Micro-bench via `dismantle bench-kernel` (added registry fixtures in
this session, see `crates/dismantle-core/src/kernel_bench.rs`).
Conditions: Claude alive, `nice -n 19`. Numbers are contended; ratios
are the useful signal.

V2-Lite expert shape (10944×2048), 1000 iter:

| Kernel                              | mean μs | p50 μs | Δ vs v3_xtg |
|-------------------------------------|---------|--------|-------------|
| `gemv_q4_k_m_v2_pinned_tcb`         | 251.1   | 204.3  | (baseline)  |
| `gemv_q4_k_m_v3_xtg_pinned`         | 436.6   | 385.3  | —           |
| `gemv_q4_k_m_v3_xtg_sumy_pinned`    | 444.1   | 395.2  | +1.7% slower |

LM head shape (102400×2048), 200 iter:

| Kernel                              | mean μs | p50 μs | Δ vs v3_xtg |
|-------------------------------------|---------|--------|-------------|
| `gemv_q4_k_m_v2_pinned_tcb`         |  549.7  |  495.0 | (baseline)  |
| `gemv_q4_k_m_v3_xtg_pinned`         | 2501.1  | 2484.6 | —           |
| `gemv_q4_k_m_v3_xtg_sumy_pinned`    | 2620.2  | 2593.1 | +4.8% slower |

**Conclusion:** the sumy trick is a regression vs `v3_xtg` at both
shapes. The extra `simd_sum` synchronization (8 per block) and the
hoisted `dm * sumy` accumulation cost more than the dm-MAD savings
recover, at least in this regime. The result is stable enough across
both shapes that a clean-window rerun is unlikely to flip the sign.

Secondary observation: `v3_xtg` itself underperforms `v2_pinned_tcb`
at both shapes under this bench (1.7× slower at expert, 4.5× slower
at LM head). That's pre-existing — `v3_xtg` ships from L7.1
(`50513c0` / `9b7038d`) but is not the active schedule in the
deployed profile (which uses `v2t_gu_v2`). The bench fixture's fresh
synthetic buffer allocation may also disadvantage the cooperative
x_cache geometry vs the warm-cache production path. Either way, the
bench reaffirms that `v2t_gu_v2` is the right default and the v3_xtg
family is currently dormant.

Action: keep the v3_xtg_sumy kernel + parity test in tree as a
documented negative result (so the next session doesn't try the same
idea blind), but **do not** flip the active profile or any per-shape
override to it. The schedule string `"v3_xtg_sumy"` remains opt-in
only.

### L7.2 (`moe_expert_pair_fused`) — bench not run

No bench fixture written for the fused kernel (would require a
matching chained-pipeline fixture for fair A/B). Parity-verified at
1e-3 rel on V2-Lite shape (2048/1408/2048) so the kernel is correct;
end-to-end win/loss measurement needs either (a) live wiring into the
routed-MoE dispatch path so `coexist_bench.sh` can measure dec_tps,
or (b) bespoke per-kernel benches for the fused vs unfused paths.
Both are deferred — the v3_xtg_sumy negative above is a useful
warning that fusion ideas need empirical validation before
integration time is spent.

## Code-vs-compute accounting (Pattern 9 reality check)

Plan estimate: ~860 LoC across 7 files, 2-3 days.

Actual landed:
- `quant.metal`: +100 LoC (new kernel)
- `kernels/mod.rs`: +99 LoC c0fc428 + 141 LoC 8073a9e = +240 LoC
- `model/deepseek_v2.rs`: +18 LoC (schedule branch for Stage 0.5)
- `metal/mod.rs`: +3 LoC (shader include)
- `shaders/moe_expert_pair_fused.metal`: +236 LoC (new file)
- `tests/q4_k_v3_xtg_sumy_parity.rs`: +163 LoC (new file)
- `tests/moe_expert_pair_fused_parity.rs`: +200 LoC (new file)
- Two profile shader_hash bumps

Total: ~960 LoC across 7 files in one session. Plan estimate was
calibrated tightly enough; the difference is the parity tests came in
a bit fatter than the budgeted ~200 LoC because two distinct test
suites were needed (one per kernel).

## Acceleration patterns applied

- **Pattern 1 (mid-flight signal):** the parity tests are the early
  signal for each kernel. Both turned green on first run, saving the
  iterative debug loop the plan budgeted for.
- **Pattern 2 (smoke ≠ eval ≠ bench):** parity (synthetic, ~1 sec) is
  not the bench. Bench deferred but its prerequisites met.
- **Pattern 8 (strip-restore):** user diagnostic +27/3 files survived
  two commits, verified at end (`git diff --stat` matches entry state).
- **Pattern 9 (code-vs-compute):** original plan's "what code is
  missing" section was load-bearing — knowing it was ~860 LoC of
  engineering kept the session honest about scope.

## Next-phase recommendation

Stage 0.5 is **resolved (negative)** — v3_xtg_sumy underperforms v3_xtg
which itself underperforms v2_pinned_tcb under this bench. No further
work on it. The kernel + parity test stay in tree as documented dead
end so the next session doesn't redo the experiment.

For L7.2 fusion the bench question is still open. Two ways to settle
it:

1. **Live-wire `moe_expert_pair_fused`** into the routed-MoE dispatch
   path so `coexist_bench.sh` can measure end-to-end dec_tps. ~150-300
   LoC of integration in `deepseek_v2.rs` (topk_gate → route arrays →
   fused dispatch). Returns the strongest signal but commits to
   integration cost before the win is confirmed.
2. **Write a chained-pipeline bench fixture** in
   `kernel_bench.rs`: a `bench_moe_expert_pair_chained` that runs
   `moe_gate_up_union_v2t` → silu_mul → `moe_down_union_v2t` against
   identical synthetic buffers, then compare to
   `bench_moe_expert_pair_fused`. ~200 LoC; gives a μs-level direct
   comparison without touching the live path. Lower commitment cost,
   weaker fidelity (synthetic buffers, no cache state).

Option 2 is the better next move for the same reason the Stage 0.5
bench was useful here — small-blast-radius measurement before
integration time.

Beyond L7:

- **Phase L5 chain-decode pipeline** is still gated on iter 5 K=4 ≥ 25%
  chain accept. L8 HALTed at 0%; the L5 prerequisite is not close to
  met until a new training iteration produces a usable chain head.
- **Phase E (tree decode)** has no prerequisites and remains the
  highest-leverage 1-2 week branch.
- **Phase F (medusa)** is still 2-4 weeks; the F.1 capture rewrite
  still needs a clean overnight window.
