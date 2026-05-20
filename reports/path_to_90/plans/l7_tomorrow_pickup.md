# L7 tomorrow pickup — direct-action plan

**Created:** 2026-05-19 late evening, while F.2 full run cooks overnight.
**Purpose:** Walk into tomorrow's L7 work with zero rediscovery cost.
**Read order:** This doc → `methodology_distilled_post_f2.md` → `phase_l7_kernel_rewrites.md`.

## First 30 min checklist

Before writing any new code:

1. **Check F.2 result.** `cat eagle4/checkpoints/medusa_v1/best_eval.json`.
   - If acceptance met (top1[0] ≥ 40% AND top1[K-1] ≥ 15%): write
     `phase_f2_done.md`, single commit (this pickup doc rides along).
   - If not met: write `phase_f2_negative.md`, single commit, then
     decide F.2 v2 vs L7 pivot. L7 path is still defensible — it
     contributes independently of F.2 success.
2. **Audit RAM headroom** for L7 bench (clean window required at end):
   `vm_stat | head -5`. L7 bench wants ≥6 GB free for V2-Lite forward.
3. **Read three reference shaders** in this order (~30 min):
   - `crates/dismantle-core/shaders/quant.metal:478` — `v3_8r` baseline
     for the new MLX Q4_K_M variant. Note the paired-nibble decode +
     sumy trick (pre-computes Σ(x_pair) before nibble loop).
   - `crates/dismantle-core/shaders/moe.metal:440-770` — the v2t /
     v2t_gu / v2t_gu_v2 / v2t_gu_v2_fc family. The L7.2 fusion target
     is the v2t_gu_v2_fc gate+up+down sequence.
   - `crates/dismantle-core/shaders/moe_union_expert.metal` — union
     pipeline kernels (already shipped `db36908`). L7.2 fusion replaces
     the chained union dispatch with a single kernel.

## L7.2 single-kernel fusion — highest leverage, smallest blast radius

**Why first:** Memory bandwidth bottleneck on MoE expert paths. Current
chained `gate → up → silu_mul → down` writes 3 intermediates to global
memory. Fusing kills the 3 writebacks. 30–50% savings on expert dispatch
projected.

**File:** `crates/dismantle-core/shaders/moe_expert_pair_fused.metal` (NEW, ~150 lines)
**Kernel name:** `moe_expert_pair_fused`
**Signature:**
```metal
kernel void moe_expert_pair_fused(
    const device uint*    route_ids       [[buffer(0)]],
    const device half*    x               [[buffer(1)]],  // hidden vector
    const device uchar*   w_gate          [[buffer(2)]],  // Q4_K_M
    const device uchar*   w_up            [[buffer(3)]],  // Q4_K_M
    const device uchar*   w_down          [[buffer(4)]],  // Q4_K_M
    device half*          y               [[buffer(5)]],  // hidden vec out
    constant uint&        routed_mid      [[buffer(6)]],
    constant uint&        hidden          [[buffer(7)]],
    threadgroup half*     x_cache         [[threadgroup(0)]],
    threadgroup half*     act_cache       [[threadgroup(1)]],
    uint                  tgid            [[threadgroup_position_in_grid]],
    uint                  tid             [[thread_position_in_threadgroup]],
    uint                  simd_lane       [[thread_index_in_simdgroup]],
    uint                  simd_id         [[simdgroup_index_in_threadgroup]]
)
```

**Threadgroup math (M3 Pro, 32 KB budget):**
- routed_mid = 1408 (V2-Lite expert dim)
- hidden = 2048 (V2-Lite hidden dim)
- `act_cache` = 1408 × sizeof(half) = **2.8 KB** (single-row act buffer)
- `x_cache` = 2048 × sizeof(half) = **4 KB** (hidden vec replica)
- Sum: **6.8 KB at 1 simdgroup/TG**. WAY under 32 KB ceiling.
- Trade-off: 1 simdgroup/TG vs v2t's 8 rows/TG. Fewer parallel TGs
  per expert dispatch, but each TG eliminates 3 global writebacks.
  Net: bench-decided per shape.

**Math layout:**
```
For one expert route:
  1. Load x[hidden] → x_cache (cooperative across simdgroup lanes)
  2. For each row r in [0..routed_mid):
     - Decode Q4_K_M block for w_gate[r,:] and w_up[r,:]
     - gate_val = Σ(w_gate[r,k] * x_cache[k])      // GEMV row
     - up_val   = Σ(w_up[r,k]   * x_cache[k])      // GEMV row
     - act_cache[r] = silu(gate_val) * up_val      // fused activation
  3. For each row h in [0..hidden):
     - Decode Q4_K_M block for w_down[h,:]
     - y[h] = Σ(w_down[h,k] * act_cache[k])        // GEMV row
```

The fusion savings: act_cache lives in threadgroup memory, never
hits global. gate_out and up_out vanish entirely (only their product
is stored, in act_cache).

## Parity test — write FIRST, then the kernel

**File:** `crates/dismantle-core/tests/l7_kernel_parity.rs` (NEW, ~200 lines)

**Why first:** Pattern 2 — synthetic parity is the cheapest decisive
signal. Write the test against the existing CHAINED pipeline output;
implement the fused kernel; gate ships on bit-identical match at fp16.

**Test shape:**
- Synthetic Q4_K_M weights (reuse `synthetic_q4_k_bytes` from
  `path_b_parity` test — same crate so import directly)
- Fixed-seed input x at hidden=2048
- Reference: `moe_routed_union_pipeline_tcb` chained dispatch
- Candidate: `moe_expert_pair_fused` single-kernel dispatch
- Tolerance: `atol=1e-3 fp16` per dismantle's CLAUDE.md verification rule

## Dispatcher wiring — BLOCKED on diagnostic edits

`crates/dismantle-core/src/kernels/mod.rs` (~150 lines) and
`crates/dismantle-core/src/model/deepseek_v2.rs` (~50 lines) are
**both diagnostic-edit-held files**. The +27 user diagnostic edits
live there. Tomorrow's first session has TWO options:

**Option A — strip-restore around the L7 wiring commit (Pattern 8):**
1. `git stash push -m "diagnostic edits" -- engine.rs kernels/mod.rs deepseek_v2.rs`
2. Write L7 wiring into `kernels/mod.rs` + `deepseek_v2.rs`
3. Build + parity test
4. Commit (L7 only, single-purpose)
5. `git stash pop`
6. Resolve any conflict between stash and new wiring (manual; the
   diagnostic edits are tracing/debug-print scaffolds and rarely
   touch the same lines as dispatcher additions)

**Option B — reconcile diagnostic edits first:**
1. Audit whether the +27 lines are still serving a purpose
2. If yes: split into a permanent commit (separate from L7)
3. If no: drop them
4. Then L7 wiring lands without strip-restore overhead

User-call decision. Option A is the dismantle-haul default per CLAUDE.md.

## Build hygiene checklist (every L7 commit)

Per project CLAUDE.md:
1. `cargo build --release --workspace` clean
2. `cargo test --workspace --lib` — pre-existing tests pass
3. New parity test passes at `atol=1e-3 fp16`
4. Regen shader_hash via `cargo run --example print_shader_hash`
5. Bump profile JSON's `shader_hash` field (Pitfall #2 from project CLAUDE.md)
6. Single-purpose commit with inline Joshua Hicks identity, no trailers
7. Diagnostic edits intact (+10 / +13 / +4 in three files)

## Why this is the right L7 entry point

`phase_l7_kernel_rewrites.md` lists three kernels:
1. `moe_expert_pair_fused` (~150 lines) — **start here**
2. `gemv_q4_k_v3_mlx` (~120 lines) — needs MLX-LM reference, do second
3. `moe_batched_gemm_q4_indexed_v3` (~180 lines) — applies #1 + #2 patterns, do third

(1) is the highest leverage because MoE expert dispatch fires
~162 invocations per token in V2-Lite (27 layers × top_k=6 routes).
Even a 30% savings here is the bulk of the L7 budget.

(2) requires external code reference (MLX-LM's `mlx_lm/models/
quantized_linear.py`). User needs to drop a copy into
`reports/path_to_90/mlx_lm_ref/` before this work. Don't block (1)
on it.

(3) reuses learnings from both — write LAST.

## Methodology pre-launch checklist for L7 (from
`methodology_distilled_post_f2.md` patterns 11–20)

- [ ] Pattern 16 — front-load: ALL THREE kernels' shader files exist
  before the first parity-test commit. Don't iterate one at a time.
- [ ] Pattern 18 — hyperparams from first principles: ROWS_PER_TG +
  simdgroup count + threadgroup memory layout derived from M3 Pro
  arch (above), NOT grid-searched.
- [ ] Pattern 2 — three validation levels defined: synthetic parity
  (~1 sec) → per-shape A/B bench (~30 sec) → clean-window full bench
  (~5 min). Each gates the next.
- [ ] Pattern 6 — bench runs ≥2× per shape for noise floor.
- [ ] Pattern 8 — strip-restore for the dispatcher commit (see
  Option A above).

## Estimated wall-clock with patterns applied

- Kernel #1 (moe_expert_pair_fused): 4–6 hr (vs plan's "1 day")
- Kernel #2 (gemv_q4_k_v3_mlx): 5–7 hr (still needs ref read)
- Kernel #3 (moe_batched_gemm_q4_indexed_v3): 4–6 hr
- Wiring + parity + clean-bench: 2–3 hr each
- **Total: ~20 hr focused work**, vs plan's 2–3 days (16–24 hr)

The savings come from patterns 16 (no mid-flight refactor) and 18
(no kernel-geometry sweep). Hyperparams above are reasoned, not
discovered.

## What L7 buys for path-to-100

Per `path_to_100_retool.md`:
- L7 realistic: +20 dec_tps → 47 dec_tps with F.2 still pending,
  ~62 dec_tps after F.2 + L7
- L7 best: +30 dec_tps → adds with F.2 best to give ~96 dec_tps
- Plus L5 (small, fast follow): ~+5 → realistic 67, best 105

L7 alone moves the realistic floor 27 → 47. If F.2 falls short
(per-head top1 doesn't cross 15% at K=7), L7 still ships value.

## NOT in this pickup (out of L7 scope)

- F.3 Rust port of medusa head — separate phase, gated on F.2 acceptance
- L5 chain-pipeline restructure — comes after L7 per retool sequence
- F.5 hybrid tree-of-medusa — stretch, gated on F.3
- Anything touching `engine.rs` (diagnostic file, ≠ dispatcher)
- Phase E revival — dead per E.0.a gate
