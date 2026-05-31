# P1 prefill-MMA — GO (validated), integration DEFERRED to an attended session

**Date:** 2026-05-31 · **Lane outcome:** the kernel is a measured win and parity-green
(agent-reported), but it is **NOT merged** — two integration blockers below make this an
attended decision, not an autonomous force-merge. Everything is preserved.

## Verdict (what the lane proved)
- **Q4_K batched GEMM via Metal simdgroup-matrix** (`gemm_q4_k_m_batched_v3w_mma`, one-simdgroup /
  8-rows tile) is **+22–24%** vs the v3w scalar path on the **tall ffn gate/up** GEMM (11008×2048,
  rows>cols), paired N=8 microbench, 3 runs, ±1% spread. It **loses** on square/wide shapes
  (ffn_down 2048×11008 −8.8%, attn q/o 2048×2048 −10–16%) — a Type-1 occupancy reality
  (ceil(rows/8) TGs underfill the M3 Pro). Hence **shape-gated to rows>cols**.
- **Parity (AGENT-REPORTED, not independently re-run by me — see "why deferred"):** synthetic Q4_K
  GEMM atol **8.0e-5 → 1.26e-4** fp16 (≈8× under the 1e-3 gate) vs both the GEMV reference and v3w;
  token-identity **PASS** (`DISMANTLE_QWEN_Q4K_MMA=1` byte-identical greedy tokens). E2E prefill TTFT
  was noise-dominated under the contaminated session (OFF spread ~17% > effect) — trust the GEMM
  microbench, re-measure TTFT clean.

## Why deferred (two blockers)
1. **Stale worktree base.** The P1 worktree branched from `22dd6f4` (eagle5 Phase C), which is **not
   on main's lineage** (it predates the `stateful/` module + the batched predec kernel). Cherry-picking
   `c9b1c07` onto current main produces a **degenerate ~1384-line misaligned conflict in
   `qwen_dense.rs`** (the decode-critical file) — unsafe to hand-resolve autonomously. The fix is a
   clean re-derivation, not a 3-way merge (recipe below).
2. **Predec mismatch (the substantive one).** P1's MMA is **v3w-layout** and was wired into the v3w
   branch. But main's shipped batched path is `if let Some(scales) = predec_cache.get(&offset) {
   predec_kernel } else { v3w_kernel }` (`qwen_dense.rs` ~6273), and **predec is default-ON**. If the
   batched predec scale-table covers the tall gate/up weights, the shipped prefill takes the **predec**
   branch → P1's MMA (in the v3w `else`) is **DORMANT exactly on the shape it wins**. The shipped TTFT
   win therefore needs the **predec-MMA twin** — which P1 built and proved parity-green but left
   **UNWIRED** ("no batched predec scale-table on this branch") — wired into the predec branch.

## Preserved artifacts (nothing lost)
- Branch **`worktree-agent-a08c1cb44eb3d4e47`** @ **`c9b1c07`** (author Joshua Hicks, unpushed); base `22dd6f4`.
- Its clean diff: `git diff 22dd6f4 c9b1c07` — 2 MMA kernels (`shaders/quant.metal` +205), 2 wrappers
  (`src/kernels/mod.rs` +172, incl. the predec-MMA twin), shape-gated wiring (`qwen_dense.rs` +49/−11),
  new `tests/q4k_batched_gemm_parity.rs` (+49), `tests/p3_batched_prefill_parity.rs` (+18).
- `stash@{0}` preserved (applied, not popped).
- Kill recorded: `reports/dead_levers.md` → "Q4_K batched MMA on rows ≤ cols shapes" (Type-1 occupancy
  + Type-2 multi-simdgroup reframe + the predec-MMA dependency).

## Integration recipe (attended)
**Option A — land the dormant v3w-MMA (low value, parity-safe).** Re-derive P1's clean diff onto main
by hand: append the 2 MMA kernels to `shaders/quant.metal`; append the 2 wrappers to `src/kernels/mod.rs`;
add `"gemm_q4_k_m_batched_v3w_mma"` to the prewarm list (~`qwen_dense.rs:1086`); add the `q4k_mma` env
flag; insert the shape-gated swap **inside the v3w `else` branch** at `qwen_dense.rs` ~6293 (and the
ffn_down v3w fallback ~6490). Add `q4k_batched_gemm_parity.rs`. Then **build + run
`q4k_batched_gemm_parity` + `p3_batched_prefill_parity` + a token-identity generate in main yourself**
before committing. Result: default-off, **dormant in shipped (predec-on)** — only fires with
`DISMANTLE_QWEN_Q4K_PREDEC=0`.

**Option B — the shipped win (do this).** In addition to A, wire the **predec-MMA twin** into the
**predec** branch (`qwen_dense.rs` ~6276) gated by `q4k_mma && rows > cols`, and confirm/build the
**batched predec scale-table coverage** for the gate/up tensors. Then a **paired prefill bench in the
shipped config** (predec ON) to confirm the +~13% TTFT survives on the real default path. This is the
only path that moves shipped TTFT; Option A alone does not.

## Anchors (current main, post draft-body merge `680cb35`)
- batched_proj Q4_K branch: `qwen_dense.rs` ~6265–6298 (predec vs v3w at ~6273).
- ffn_down batched: ~6476–6497. Prewarm list: ~1086. predec_active default-on: ~4688.
