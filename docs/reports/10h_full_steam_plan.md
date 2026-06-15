# 10-hour full-steam production plan

Started: 2026-05-22 around 17:00 local while eagle5_train (PID 50520) is in epoch ~1-2.

## Concurrent phase (now → T+3.5h)

Eagle5 training (already running, MPS/MLX, ~5 GB RAM, ~90% CPU) and CPU-bound development work do NOT compete for the same resources. CPU dev work can land during the training window.

### A. Minimal-corpus extraction script optimization (now, ~15 min code, 0 compute)
Upgrade `tools/training/build_minimal_corpus.py` to use `ProcessPoolExecutor` for parallel shard-reads + writes. 30-min sequential → ~5 min parallel.

### B. Mixed-precision dispatcher wedge implementation (now → T+3h, mostly CPU dev)
**The gap:** `MixedQuantStore` + `TierMap` are landed in `crates/dismantle-core/src/`, but `model/deepseek_v2.rs` still does `tensor_ref(&gguf, "ffn_gate_exps.weight")` — never invokes the mixed store.

#### Sub-steps
1. **Engine plumbing:** ensure `EngineConfig.quant_tier_map_path` flows to model construction (~30 min).
2. **Load + build MixedQuantStore:** if `quant_tier_map_path.is_some()`, load TierMap, dequant fp16 each layer's expert weights, requantize per the tier (~60 min).
3. **Tensor lookup swap:** replace direct GGUF tensor refs for expert weights with `MixedQuantStore.lookup(layer, kind)` returning `(blob_slice, dtype)` (~60 min).
4. **Kernel dispatcher:** for each layer's selected dtype, route to `q4_K`/`q6_K`/`q8_0` kernel. Existing `kernels/mod.rs` has the dispatch table — extend it to read from the per-layer dtype (~30 min).
5. **`cargo build --release` + unit test** (~5 min compute).

Defer the cargo build until eagle5 finishes (cargo build is CPU-heavy and would compete).

## Pipeline-finish phase (T+3.5h → T+4.75h)

Pipeline auto-runs the remaining stages:
- τ-eval (~30 min)
- q4 quantize + parity (~15 min)
- vocab-prune paired bench (~30 min) → **real L1 dec_tps delta**

## Post-pipeline phase (T+4.75h → T+5.0h, ~15 min)

1. Run parallel `build_minimal_corpus.py`. 80 GB → ~4 GB.
2. `rm -rf artifacts/calibration/v2_lite_corpus` (original). **+76 GB freed.**
3. Verify eagle5 reads the minimal corpus correctly (quick smoke).

## Mixed-precision finalize phase (T+5.0h → T+7.5h, ~2.5h)

1. `cargo build --release -p dismantle` with the new code (~2 min).
2. **Parity test**: mixed-precision vs uniform fp16, fixed prompt + seed, 64 tokens. Bit-identical greedy required (~5 min).
3. **Microbench**: per-layer GEMM with each bit-width — verify expected speedup (~10 min).
4. **End-to-end bench**: dec_tps with `--quant-tier-map-path` set vs unset (~15 min).
5. **Iterate tier map** if numbers undershoot:
   - Try more aggressive q4 distribution (layers 0-5 → q4 instead of just 0-3)
   - Re-run parity + bench (~20 min per iteration)
   Estimated 2-3 iterations to land target.

Total compute: ~1.5h. Total wall-clock including iteration: ~2.5h.

## Spec-decode runtime investigation phase (T+7.5h → T+10h, ~2.5h)

The K=1 diagnostic returned exit 2 — eagle4 K=1 spec-decode fails all trials. Eagle5 head can't deliver tps until this is fixed. Investigate root cause.

1. Re-run K=1 diagnostic with verbose tracing (`DISMANTLE_TRACE_DISPATCH=1`) to see where it fails (~15 min).
2. Read `crates/dismantle-core/src/speculate/` end-to-end (~45 min).
3. Identify failure mode — likely candidates per `path_to_100_repath` memory:
   - Verifier/draft sequence mismatch
   - Kernel-not-found in some bit-width combo
   - Cache pollution between target and draft passes
4. Attempt targeted fix (~60 min code).
5. Re-run diagnostic. If exit 0: success. If still failure: document findings for next session (~15 min).

## What we expect to land by T+10h

| Deliverable | Confidence | Tps impact |
|---|---|---|
| Eagle5 v2 head trained, τ-eval'd, q4 quantized | High | 0 until spec-decode lands |
| Vocab-prune real dec_tps delta measured | High | +1-3 |
| Mixed-precision Path A wired + benched | High | +5-10 |
| Minimal corpus, 76 GB freed | High | n/a |
| Spec-decode runtime debugged | Medium | +5-15 if fixed |

**Realistic outcome:** ~30-37 dec_tps if mixed-precision lands at projection. ~42-50 if spec-decode also lands.

**Pessimistic outcome:** ~30 if mixed-precision needs more parity iterations and spec-decode stays broken.

## Risk register

| Risk | Mitigation |
|---|---|
| Eagle5 train OOMs as RSS climbed to 8.7 GB earlier | Pipeline will auto-fail-and-continue to bench |
| Cargo build competes with eagle5 if started too early | Defer cargo build until eagle5 done |
| Mixed-precision parity fails on first try (kernel-not-found) | Iterate tier map — fallback to fewer q4 layers |
| Spec-decode root cause is non-trivial | Document for next session; don't burn compute on guesses |

## Optimization principles applied
- CPU dev work in parallel with GPU train (no resource competition)
- Cargo builds gated on eagle5 done (CPU competition)
- Compute-heavy steps after eagle5 finishes (free up resources)
- Idempotent: every script can re-run on failure without re-paying setup cost
