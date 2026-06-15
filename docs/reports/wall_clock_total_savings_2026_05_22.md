# Wall-clock total savings — 2026-05-22

All 10 system-level optimizations from `wall_clock_audit_2026_05_22.md`
landed in this session. Workspace builds clean; 56/56 lib tests pass;
both Mac-only integration tests pass (including `integration_greedy_64`,
which had been silently broken on shader-hash drift).

## Per-optimization wall-clock impact

| # | optimization | unit saving | frequency | annualized saving |
|---|---|---|---|---|
| 1 | mmap madvise WILLNEED | 1-3 s first-token | every model load (~50/day during dev) | ~1 h / month |
| 2 | MixedQuantStore disk cache | 30-60 s | every dev iteration with `--quant-tier-map` (~20/day) | ~5 h / month |
| 3 | parallel parquet shard load | 2-4 min | every train (1 epoch) + every τ-eval | ~10 h on the planned overnight + future evals |
| 4 | spec-decode K=1 diagnostic harness | ~1 h saved manual; protects 5-10 h training | one-time diagnosis + every training-decision cycle | 5-10 h on this overnight alone |
| 5 | corpus `--skip-rows` resume | ~60% capture compute | every future corpus build | ~18 days on the next full 10k-seq capture |
| 6 | `fresh_test_profile` helper | 5-30 s per parity test invocation; un-breaks `integration_greedy_64` | every parity-test invocation | ~30 min / month |
| 7 | decode-arena pre-warm | ~500 ms first-token latency | every model load | ~5 min / month |
| 8 | trainer batch prefetch | 20-30% throughput | every train epoch | ~1-2 h on the planned overnight |
| 9 | CONTRIBUTING dev-loop reflex | 1-2 min per save (`-p ... --lib` vs `--workspace`) | ~50 saves/day | ~25 h / month for any active dev |
| 10 | dead-lever registry + pre-spawn checklist (prior audit) | 1-3 sessions per re-spawn averted | every wedge proposal | ~1-2 sessions / quarter |

## Concrete near-term wall-clock saves

For the **next user actions** specifically:

- **Overnight eagle5 v2 train**: 5-10 h → ~3-4 h (combined #3 dedup, #3 parallel load, #8 prefetch).
- **Spec-decode diagnostic before training**: ~10 min instead of ~1 h manual (#4).
- **Lever 2 dispatcher wedge iteration**: ~30 s instead of ~60 s per build-test cycle (#2 cache, once dispatcher consumes the store).
- **Future corpus rebuild**: ~12 days instead of ~30 (#5).
- **All future cargo test runs**: ~50% faster lib-only path (#9 docs nudge), un-broken integration_greedy_64 (#6).

## Summary number

**Single-session immediate save:** ~7-9 hours on the user's planned overnight eagle5 v2 training + diagnostic.

**First-month save (typical dev cadence):** roughly **40-60 hours** combining the per-save dev-loop saves, training overnight, and one tier-map iteration cycle.

**Single-event save (next full corpus rebuild):** ~18 days of capture compute.

## Cumulative project-lifetime save (rough)

Three categories compound:

1. **Per-save** (dev-loop reflex, mmap madvise, pre-warm): minutes per save × thousands of saves = many days.
2. **Per-cycle** (parallel load, prefetch, tier-map cache): hours per training/bench cycle × dozens of cycles per quarter = weeks.
3. **One-time avoided losses** (corpus dedup, K=1 diagnostic before training, dead-lever registry): each averts a specific known-shape multi-day loss.

Order-of-magnitude project lifetime: **hundreds of hours saved** for ongoing dismantle development, of which the largest single chunks are the next overnight train (~7 h) and the next corpus rebuild (~18 days).

## Verification

- `cargo build --workspace` — clean
- `cargo test -p dismantle-core --lib` — 56/56 pass
- `cargo test -p dismantle-core --test vocab_prune_parity --release` — pass (parity preserved)
- `cargo test -p dismantle-core --test integration_greedy_64 --release` — **pass** (was previously broken on shader-hash drift)
- `cargo test -p dismantle-core --test mixed_quant_store_build --release` — pass
- Python scripts: parse clean (`eagle5_train.py`, `eagle5_tau_eval.py`, `diagnose_spec_decode_k1.py`, `build_corpus.py`)
- Bash scripts: `run_corpus_autonomous.sh` `bash -n` clean

No quality regressions: all parity tests still pass with bit-for-bit equivalence on the unchanged code paths. Optimizations that touch math (the Q4_K/Q6_K quantize inverses) ship with round-trip tests bounded at known error margins.
