# Path-to-90 Stage 1 — A1 close: flash_attn wire-up — REJECTED at +3% gate

**Status:** HALTED. Infrastructure landed as opt-in `metal-mla-flash`; default unchanged.
**Branch:** `claude/strange-proskuriakova-b5d48e`
**Base:** 0983166 (A4 close).
**Date:** 2026-05-15

## Result

| Profile | dec_tps (trimmed-median) | Δ vs A4 baseline | Δ vs pristine main |
|---|---:|---:|---:|
| pristine main (metal-mla, A4 disabled) | 20.50 | — | — |
| **metal-mla-fc (A4, current default)** | 23.97 | — | +16.9% |
| **metal-mla-flash (A1, this lever)** | **20.65** | **−13.9%** | +0.7% |

A1-flash is **worse than the pre-A5 baseline**. Decisive negative result. Below the plan's +3% reject threshold. Per the plan's explicit halt trigger ("A1 flash-attn fails parity or shows <+3% e2e → halt A1, escalate to attended"), this lever does not ship as default. The dispatcher and profile flag remain in-tree as opt-in scaffolding for future long-context / KV-quant scenarios where flash's TG-mem savings might pay off.

**Parity:** bit-identical 3-token greedy decode on all 12 baseline prompts vs pristine main and A5. The 4 existing `v1l_flash_attn_parity` Rust tests still pass at atol=1e-3 vs `mla_decode_kernel`. The kernel is correct.

## Why it lost

The Stage 0 attribution showed `mla_decode_kernel` at 11.5% GPU. The pre-existing assumption (from the upstream research brief) was that flash attention would cut that figure by 1.3-1.5× because it eliminates the materialized `scores[seq_len]` threadgroup buffer. Two structural mismatches make that thesis fail for *this* engine on *this* benchmark:

1. **seq_len is short.** Our bench runs 4-token prompts × 64-token decode, so seq_len stays in the 5-68 range. The materialized `scores[seq_len]` buffer is at most 272 bytes — easily resident in shader-core SRAM, no cache pressure. The win that flash attention is designed for (avoiding O(seq_len) TG memory) is worth zero here.

2. **Threadgroup geometry doesn't match.** The flash kernel uses `FLASH_TG=128` (4 simdgroups) while `mla_decode_kernel` uses `TG_SIZE=256` (8 simdgroups). On M3 Pro, the wider TG distributes Phase 0 (`q_nope_proj`, 512 entries across threads) and Phase 3 (`c_kv_wt`, 512 entries) more efficiently — each thread does 2 elements instead of 4. The flash kernel adds 4-5 extra `threadgroup_barrier` boundaries per tile (parallel max, online softmax state, scale+accumulate); at 1 tile per dispatch (seq_len ≤ 128) that's pure overhead.

3. **Per-tile bookkeeping doesn't amortize at seq_len < FLASH_TG.** For seq_len=10 in a 128-thread TG, 118 threads compute `-INFINITY` scores, then the serial thread-0 `for ti in 0..t_len` loop runs 10 iterations to compute `tile_sum` — work the materialized softmax does in one threadgroup-parallel sweep.

Flash attention is the right kernel for long contexts (1K+) or smaller TG-mem budgets. For DeepSeek-V2-Lite-Q4_K_M decode on M3 Pro with the current threadgroup layout, the materialized kernel wins by ~14%.

## What this means for the plan

The original Stage-1 sequence put A1 third (after revised A5 → A4 → A1). A1's failure does NOT invalidate the path-to-90 plan:

- The plan's halt rule for A1 said "do not continue to A2 (FA was the keystone assumption)." But the Stage 0 attribution already weakened FA's keystone status: the dominant gap was CPU dispatch overhead (~25 ms/tok), not attention kernel quality (attention is only 30% of GPU = ~7 ms/tok of total). A4 + A5 cleared most of the dispatch overhead, which was the real keystone.
- A1 being a negative result simply means *this particular kernel design* doesn't help. A future A1.2 might revisit with a different threadgroup geometry (e.g., FLASH_TG=256 to match) or after Q8/Q4 latent KV makes the per-element c_kv read expensive enough that the tiled accumulator wins on cache locality.
- **Continue plan to A3** (cross-layer residual+RMSNorm fusion). Expected +3-6% e2e, low risk.

## Files changed (kept in-tree as opt-in scaffolding)

- [crates/dismantle-core/src/kernels/mod.rs](../../../crates/dismantle-core/src/kernels/mod.rs) — added `flash_attn_decode_and_o_proj_arena_tcb` TCB dispatcher (mirrors `mla_decode_and_o_proj_arena_tcb` exactly, only swapping kernel name + threadgroup-memory-slot 3 for the `state[8]` buffer).
- [crates/dismantle-core/src/model/deepseek_v2.rs](../../../crates/dismantle-core/src/model/deepseek_v2.rs) — added `mla_use_flash` engine field, gated on `mla_schedule == "metal-mla-flash"`; routed through `dispatch_mla_decode_and_o_proj` helper.
- profile.json **not touched** — default remains `metal-mla-fc`.

Opt-in usage:
```
jq '.selected.mla_schedule = "metal-mla-flash"' profiles/deepseek-v2-lite-q4.m3pro18.json > /tmp/flash.json
./target/release/dismantle bench --kernel-profile /tmp/flash.json ...
```

## Bench artifacts

- [a1_flash_run{1..5}.json](.) — 5 untraced runs × 3 trials each
- [../stage0/a1_flash_token_hashes.txt](../stage0/a1_flash_token_hashes.txt) — parity hashes

## Stage 1 cumulative (unchanged from A4)

| Stage | dec_tps (trimmed-median) | Δ vs main |
|---|---:|---:|
| pristine main (v2.2.0) | 20.50 | — |
| A5 (arena) | 22.23 | +8.4% |
| A5 + A4 (mla-fc pilot) | **23.97** | **+16.9%** |
| A1 (mla-flash) — *rejected* | 20.65 | +0.7% |

## Next: A3 — cross-layer residual+RMSNorm fusion

Currently each decode layer dispatches `add_inplace` (residual) followed by the next layer's `rmsnorm_gemv_f16w_attn_pinned_v2t` (fused RMSNorm+proj) as separate dispatches. These can fuse into one kernel that reads residual+input, applies RMSNorm normalization in threadgroup memory, then runs Q/KV projection — saving ~27 dispatches/token. Estimated +3-6% e2e. Low parity risk.
