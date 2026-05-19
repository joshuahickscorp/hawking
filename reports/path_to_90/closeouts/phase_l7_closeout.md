# Phase L7 closeout — kernel rewrites (Stage 0.5 + L7.2)

**Status:** SHIPPED kernels + parity gates. Live wiring + clean-window
bench deferred to follow-on session.

**Branch:** `claude/dreamy-golick-d54ff8`
**Commits:**
- `c0fc428` — `path-to-150 L7 / Stage 0.5 — gemm_q4_k_m_v3_xtg_sumy kernel`
- `8073a9e` — `path-to-150 L7.2 — mixed-quant moe_expert_pair_fused kernel + parity`

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

**TBD — bench queued.** Both kernels are parity-verified but not
benched. Stage 0.5 targets the LM head shape (102400×2048) where x is
redundantly read per-TG; expected single-digit % win on the LM head
path. L7.2 targets MoE expert dispatch elimination of intermediate
buffer traffic; expected K=1 win, ambiguous at K=4 (loses union expert
reuse).

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

Either of two paths makes sense depending on what the next session
wants to optimize for:

1. **Bench-first.** Run `tools/bench/path_to_125_bench.sh` against
   `gemm_q4_k_schedule = "v3_xtg_sumy"` on the LM head path. If it
   wins ≥5%, ship the per-shape profile bump. Then attempt live wiring
   for `moe_expert_pair_fused` and bench that. Lowest-risk path; turns
   the parity-verified kernels into actual dec_tps.

2. **Phase L5 chain-decode pipeline.** With L8 training HALTed at 0%
   chain accept, the chain-decode pipeline is currently moot — the
   prerequisite "iter 5 K=4 ≥ 25%" is not even close to met. L5 should
   wait for a successful K≥4 chain-accept training run from a new L8
   iteration.

Phase E (tree decode) remains a viable 1-2 week branch with no
prerequisites; can be started in parallel with the L7 bench work.
