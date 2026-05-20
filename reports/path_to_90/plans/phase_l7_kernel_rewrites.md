# Phase L7 — kernel rewrites (L7.2 fusion + Stage 0.5 MLX family)

**Goal:** +15-30 dec_tps. Combines L7.2 single-kernel fusion +
broader MLX-style kernel rewrites (Stage 0.5 from prompt).
**Engineering est:** 2-3 days focused work.
**Confidence:** MEDIUM — kernel work is empirical; each shader needs
iterative tuning.

**Prereq reading:** `acceleration_patterns.md` (patterns 1–10) +
`methodology_distilled_post_f2.md` (patterns 11–20) +
`path_to_100_retool.md` (this phase contributes +20 realistic to the
100-tps target). Apply the 20/20 pre-launch checklist before starting.
Most-relevant patterns for L7: **16** (front-load all kernel patterns
into one infra commit), **18** (kernel geometry reasoned from M3 Pro
arch, not grid-searched), **2** (synthetic parity → clean-window
bench is the decisive gate, not mid-flight intuition).

## What code is missing (Pattern 9)

| File | Lines (est) | Purpose |
|---|---|---|
| `crates/dismantle-core/shaders/moe_expert_pair_fused.metal` | ~150 | L7.2 single-kernel gate+up+silu+down |
| `crates/dismantle-core/shaders/gemv_q4_k_v3_mlx.metal` | ~120 | New MLX-LM-style Q4_K_M GEMV (broader than xtg) |
| `crates/dismantle-core/shaders/moe_batched_gemm_q4_indexed_v3.metal` | ~180 | MoE Q4_K_M GEMV with MLX patterns |
| `crates/dismantle-core/src/kernels/mod.rs` | ~150 | Three new dispatchers + tracing names |
| `crates/dismantle-core/src/model/deepseek_v2.rs` | ~50 | Schedule branches for the new kernels |
| `crates/dismantle-core/src/profile.rs` | ~10 | Two new schedule string values |
| `crates/dismantle-core/tests/l7_kernel_parity.rs` | ~200 | Parity gates for all three new kernels |

**Net:** ~860 lines across 7 files. Mostly new shader code + parity
test scaffolding.

## Why this lever

The current Q4_K_M kernel stack has been heavily optimized for the
8-rows-per-TG geometry used by v3_8r. But L7.1 found a real gap: the
standalone GEMV doesn't use threadgroup x_cache (8× redundant device
reads per TG). xtg closed that gap for the standalone path.

Two more kernel-level optimizations remain:

1. **L7.2 single-kernel fusion** — gate + up + silu_mul + down in one
   Metal kernel. Eliminates 3 intermediate writes to global memory
   (gate_out, up_out, act). Memory bandwidth is the bottleneck on
   MoE expert paths; eliminating 3× the writeback saves 30-50% on
   the expert dispatch.

2. **MLX-LM Q4_K_M GEMV pattern** — MLX-LM's `mlx_lm/models/deepseek_v2.py`
   uses a specific simd-shuffle + paired-nibble pattern that
   outperforms the current v3 stack on certain shapes. Worth porting.

## What's already shipped

- xtg standalone Q4_K_M kernel + parity + schedule wire (commits
  `50513c0` and `9b7038d`)
- shader_hash regen helper (`crates/dismantle-core/examples/print_shader_hash.rs`)

## Concrete plan

### Step 1 — Read existing kernels (1 hr)

Files to study:
- `crates/dismantle-core/shaders/quant.metal` line 478 — v3_8r, the
  baseline for the new MLX variant
- `crates/dismantle-core/shaders/moe.metal` line 440-770 — v2t,
  v2t_gu, v2t_gu_v2, v2t_gu_v2_fc — the current MoE expert pipeline
- `crates/dismantle-core/shaders/moe_union_expert.metal` — the
  union pipeline kernels (already shipped in `db36908`)

The L7.2 fusion target is the union pipeline's gate_up + down sequence.

### Step 2 — L7.2 single-kernel fusion shader (1 day)

`shaders/moe_expert_pair_fused.metal::moe_expert_pair_fused`:
- Input: `route_ids`, `x` (hidden vector), `w_gate`, `w_up`, `w_down`
- Output: `y` (hidden vector contribution per route)
- Internal: gate(x), up(x), act = silu(gate)*up, y = down(act)
- Geometry: per-route TG; threadgroup memory holds x_cache + act_cache
- Register/shmem math: at routed_mid=1408, act_cache = 5.6 KB; x_cache
  = 8 KB; sum 13.6 KB at 1 simdgroup/TG. Within M3 Pro's 32KB budget.
- Trade-off: 1 simdgroup/TG vs v2's 8 rows/TG. Need to bench whether
  fusion savings outweigh TG count reduction.

Parity: against the chained `moe_routed_union_pipeline_tcb` output
at `atol=1e-3 fp16` on synthetic Q4_K_M weights.

### Step 3 — MLX-LM Q4_K_M GEMV (1 day)

`shaders/gemv_q4_k_v3_mlx.metal::gemv_q4_k_v3_mlx`:

The MLX-LM pattern (paraphrased from their public deepseek_v2.py
kernel):
- N_R0 = 1 row per simdgroup (not 8 like v3_8r)
- N_SIMDGROUPS = 8 per TG
- simd_shuffle for cross-lane reduction
- Paired-nibble decode (one byte = 2 nibbles, like v3_8r)
- Sumy trick: pre-compute Σ(x_pair) before nibble loop to halve mults

Key file to reference: MLX-LM repo's quant ops in
`mlx_lm/models/quantized_linear.py` (needs cross-repo read; the user
can drop a copy into reports/path_to_90/mlx_lm_ref.py for in-repo
reference).

Parity: against CPU dequant + gemv_f32 at `atol=1e-3 fp16` on the
LM head shape (102400 × 2048) and V2-Lite expert shape (10944 × 2048).

### Step 4 — Broader MoE Q4_K_M v3 (1 day)

`shaders/moe_batched_gemm_q4_indexed_v3.metal`:

Apply L7.1's threadgroup-x_cache + L7.2's fusion + MLX patterns to
the MoE batched GEMV (`moe_batched_gemm_q4_indexed_v2t_gu_v2_fc`
replacement).

This is the highest-leverage shader because MoE expert dispatch fires
27 layers × top_k=6 routes per token = ~162 invocations per token
in V2-Lite.

Parity: against `moe_batched_gemm_q4_indexed_v2t_gu_v2_fc` at
`atol=1e-3 fp16`.

### Step 5 — Schedule wiring (1 hr)

Add to `crates/dismantle-core/src/profile.rs`:
- `gemm_q4_k_schedule = "v3_mlx"` (selects gemv_q4_k_v3_mlx)
- `moe_schedule = "expert_pair_fused"` (selects moe_expert_pair_fused)
- `moe_schedule = "v3_mlx"` (selects moe_batched_gemm_q4_indexed_v3)

Add corresponding branches in `crates/dismantle-core/src/model/deepseek_v2.rs`
at the existing schedule selectors.

### Step 6 — Parity tests (2 hr)

`tests/l7_kernel_parity.rs`:
- Synthetic Q4_K_M weights (uses path_b_parity's `synthetic_q4_k_bytes`)
- For each new kernel, dispatch and compare against CPU ref
- Tolerance: 1e-5 relative (per q4_k_v3_xtg_parity.rs convention)

### Step 7 — Shader hash + clean bench (1 hr)

- Regenerate shader_hash via `cargo run --example print_shader_hash`
- Bump profile JSON's `shader_hash` field
- Cmd-Q Claude, run `tools/bench/path_to_125_bench.sh` with each new
  schedule enabled per-shape via `gemm_q4_k_schedule_per_shape` /
  `moe_schedule_per_shape`
- Per-shape A/B: keep new kernel ON only if ≥5% wall improvement at
  the shape tested

## Risks + mitigations

1. **New shader has subtle bug, fp16 quantization noise hides it.**
   *Mitigation:* parity test uses synthetic exact weights; deterministic
   inputs catch math errors. Tolerance ≤ 1e-5 relative.

2. **MLX-LM pattern doesn't transfer (different GPU, different shape).**
   *Mitigation:* keep old kernels in place; new ones opt-in via
   profile flag. No regression risk.

3. **Threadgroup memory budget exceeded on M3 Pro.**
   *Mitigation:* 32 KB ceiling validated upfront. If budget tight,
   reduce ROWS_PER_TG to leave room (already done for xtg pattern).

4. **shader_hash drift breaks existing profiles.**
   *Mitigation:* regenerate hash in same commit as kernel landing.
   Pitfall #2 from project CLAUDE.md.

## Acceleration patterns applied

- Pattern 1: each new kernel gets a parity test (mid-flight signal)
- Pattern 2: synthetic parity + clean bench levels defined
- Pattern 6: bench runs 10 trials per shape
- Pattern 7: kernel-rewrite tasks can run in parallel (independent shaders);
  use autoiter to sweep them
- Pattern 9: "code missing" lists ~860 lines across 7 files

## Acceptance criteria

- All three new kernels pass parity at `atol=1e-3 fp16` / 1e-5 rel
- At least one of the three shows ≥5% bench improvement on V2-Lite
  shape (expert_pair_fused on 1408×2048 most likely)
- Existing parity gates (`path_b_parity`, `eagle4_decode_parity`)
  still pass

## Next-session quickstart

```
# 1. Read this + acceleration_patterns.md
# 2. Get MLX-LM source — clone https://github.com/ml-explore/mlx-lm
#    or copy ref kernels to reports/path_to_90/mlx_lm_ref/
# 3. Start with L7.2 single-kernel fusion (highest leverage, smallest
#    blast radius)
# 4. Parity test BEFORE benching
# 5. Bench in clean window after parity green
```
