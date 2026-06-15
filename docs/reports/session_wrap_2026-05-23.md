# Session wrap — 2026-05-23

Comprehensive synthesis of what happened in this multi-day session. Read
this first when picking up.

## TL;DR

- **Production tps delivered:** baseline 24.88 → L1 vocab-prune 26.43.
  **+1.55 dec_tps (+6.2%) reliable.** 30-trial variance hunt: stdev 0.12,
  95% CI ±0.04.
- **Path to 50 remains gated** on real Q8 KV wiring (currently broken),
  MoE GEMM kernel work, and operator fusion.
- **Path to 75 needs ~4-6 weeks of focused Metal/Rust work.** Today's
  autonomous work has produced the *foundation + diagnostics* but
  cannot itself reach 75.

## Confirmed deployable today (single lever)

| Lever | Δ vs baseline | Variance | Files |
|---|---:|---:|---|
| **L1 vocab-prune** (`--vocab-prune-path artifacts/calibration/analysis/vocab_whitelist_995.json`) | **+1.55 tps** | σ=0.12 | wired in main |

Other levers measured-as-marginal-or-broken this session:

| Lever | Status |
|---|---|
| L2 mixed-precision (tier_default / tier_aggro) | +0.5-2.5 tps; **does not compound with L1** (M1 stack matrix) |
| L4 n-gram drafts | +0 net when L1 already on (M7) — drop from STACK |
| L4 exact-shared spec-decode (Session B) | broken at K≥8 even after K-revert; revert holds the line |
| L3 Q8 KV (Session C) | **patch broken — flag still missing after rebuild attempt** |
| MLA Phase 4 (D) | DEAD — -1.7% to -2.5% measured |
| CPU+GPU pipelining (I) | DEAD per memory `cpu_gpu_pipelining_audit.md` |
| MoE GEMM Q8_0 w2 (J) | landed in main env-gated, +1.33% single-shape |
| RMSNorm fusion (F) | 1/6 sites wired, flat measured |

## 149 dirty files — categorized

### Group 1 — Untracked NEW files (keep, add to git)
```
crates/dismantle-core/src/mixed_quant_store.rs           Session A/C foundation
crates/dismantle-core/src/quant_tier_map.rs              Session A
crates/dismantle-core/src/vocab_prune.rs                 Session A — IN USE today
crates/dismantle-core/tests/mixed_quant_store_build.rs   Session A parity
crates/dismantle-core/tests/q8_kv_parity.rs              Session C parity (passes)
crates/dismantle-core/tests/vocab_prune_parity.rs        Session A parity
crates/dismantle-core/tests/mla_flash_v2lite_parity.rs   MLA work
crates/dismantle-core/tests/mla_q8kv_microbench.rs       Q8 KV bench
crates/dismantle-core/tests/rmsnorm_fused_parity.rs      Session F parity
tools/bench/*.sh                                          all 7 bench scripts I wrote
tools/training/*                                          chains + corpus tooling
eagle4/                                                   prior eagle work (separate decision)
```

### Group 2 — Modified Rust src (per-hunk review)
| File | What changed | Disposition |
|---|---|---|
| `crates/dismantle-core/src/lib.rs` | module exports for new modules | **commit** with new modules |
| `crates/dismantle-core/src/engine.rs` | `vocab_prune_path`, `quant_tier_map_path` fields | **commit** — these are LIVE flags |
| `crates/dismantle-core/src/model/deepseek_v2.rs` | K revert + J w2 + arena bump 8→17 | **commit K + J w2**; verify arena is intentional |
| `crates/dismantle-core/src/kernels/mod.rs` | J w2 kernel additions + Q8 KV kernels | **commit** kernels |
| `crates/dismantle-core/src/metal/decode_arena.rs` | max_batch=17 (Session B arena) | review — kept after K revert? |
| `crates/dismantle-core/src/cache/{mod,prefill_disk}.rs` | KV cache changes | **review** — Q8 KV ambiguous state |
| `crates/dismantle-core/src/attn/mod.rs` | MLA Phase 4 cherry-pick? | **review** — DEAD per measurement |
| `crates/dismantle-core/src/moe/{mod,dispatch}.rs` | tier-map dispatch wiring | **commit** for L2 to function |

### Group 3 — Modified tests (39)
Mostly existing parity tests with minor adjustments. **Commit alongside the Rust changes they validate.** Spot-check that none have been weakened.

### Group 4 — reports/archive deletions (19)
Moved-out historical reports. **Defer** — manual decision per project hygiene.

### Group 5 — top-level (.gitignore, README, etc)
Minor housekeeping. **Commit** in a "chore" commit.

## Recommended commit sequence (no Claude attribution)

```
1. chore: add vocab_prune, quant_tier_map, mixed_quant_store modules
   - 3 new core src files + lib.rs export
   - 3 new tests (vocab_prune_parity, mixed_quant_store_build, q8_kv_parity)
   - engine.rs flag fields

2. spec-decode: revert batched-verify regression (+2-4 tps)
   - deepseek_v2.rs K hunks (forward_token_argmax restored)
   - Keep DecodeArena.max_batch_size=17 for future work
   - Measured: L0 +2.18 tps, L4_K4 +4.65 tps

3. kernel: q8_0_v3 interleaved 2-way batching for MoE down (+1.33%)
   - kernels/mod.rs J w2 hunks
   - env-gated; opt-in only

4. infra: bench harness + path-to-50 chain scripts
   - tools/bench/* (microbench_levers, pause/resume, path_to_50_matrix,
     spec_decode_sweep, phase1_go_no_go)
   - tools/training/* (chains, corpus tooling)

5. defer: Q8 KV wiring (patch broken; needs full re-investigation)
   - DO NOT COMMIT cache/ + attn/ Q8-KV hunks until wiring proves functional
   - Patch sits at reports/patches/session_C_q8_kv_wiring.patch

6. defer: MLA Phase 4 (measured DEAD)
   - revert any attn/ MLA Phase 4 hunks
   - leave on claude/mla-phase4-experiment branch as record
```

## Open issues for next session

1. **Q8 KV patch is broken** — applies (or "already applied") but `--q8-kv`
   flag doesn't surface. Needs end-to-end debug: where does the patch
   add the flag? Did the chain's `git apply --check` lie about
   already-applied? Verify the actual CLI registration.
2. **149 dirty files in main** — bench numbers are technically against
   an unstable working tree. Should commit the safe sequence above
   before any decision-grade bench.
3. **Variance noise** — even L1 alone showed 26.10-26.64 range across
   trials. 30-trial M9 nailed it to ±0.04 but cheaper modules (TRIALS=5)
   showed 1-2 tps spread. **Future benches default to TRIALS≥15.**
4. **MoE GEMM is the biggest unexplored lever** — 50.5% of decode time
   per `per_kernel_time_breakdown.md`. Needs week-scale focused work.

## Memory updates needed
- `path_to_50_complete.md` superseded — L1 alone is the deployable; L2/L4
  do not compound; Q8 KV still not wired
- Add new memory: `session_wrap_2026_05_23.md` pointing at this report
