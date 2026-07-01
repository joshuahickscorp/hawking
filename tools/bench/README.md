# tools/bench/

Benchmark harness for dismantle. All numbers are reproducible from this directory.

## Which tool to use

| situation | tool | params | time |
|---|---|---|---|
| dev iteration | `coexist_bench.sh` | TRIALS=4 TOKENS=24 | ~3–5 min |
| sub-phase commit gate | `coexist_bench.sh` | TRIALS=4 TOKENS=24 | ~3–5 min |
| phase close bench | `coexist_bench.sh` | TRIALS=6 TOKENS=64 | ~15 min |
| authoritative ship | `clean_bench.sh` | TRIALS=10 TOKENS=64 | ~25 min |
| per-kernel comparison | `dismantle bench-kernel` | `--iterations 500` | <30 s |
| cross-commit decision | `bench_diff.sh` | HEAD~1 HEAD | instant |

**For authoritative ship numbers: quit the agent app first, then run `clean_bench.sh` from a fresh terminal.** A running agent session inflates dec_tps 4–5×; the inflation cancels in paired A/B deltas but contaminates absolute numbers.

## Standardized parameters (do not change per session)

```
dev / commit gate:  TRIALS=4 TOKENS=24
phase close:        TRIALS=6 TOKENS=64
authoritative ship: TRIALS=10 TOKENS=64  (clean_bench.sh)
```

Ad-hoc params make cross-commit comparisons unreliable.

## Reading CI / IQR / outlier flags

`coexist_bench.sh` reports:
```
median: 19.8 dec_tps (95% CI: [18.9, 20.7], IQR: 1.4)
```

- **median**: 50th percentile of valid trials. Primary metric.
- **95% CI**: normal approximation `median ± 1.96σ/√N`. Overlapping CIs = no significant difference.
- **IQR**: interquartile range (Q3 − Q1). `IQR/median > 15%` triggers `⚠ SPREAD HIGH`.
- **trimmed_mean**: median with 25% dropped from each tail (N≥4). Robust to a single anomalous trial.

`⚠ SPREAD HIGH` means the agent app was probably doing heavy GPU work during some trials. Discard and re-run with more settling time, or note it as a coexist-mode limitation.

## bench_diff.sh — cross-commit significance test

```sh
bash tools/bench/bench_diff.sh HEAD~1 HEAD
```

Reads `bench_results/bench_history.jsonl` (written by every coexist_bench run). Reports both commits' median + 95% CI, delta %, and whether the difference is statistically significant (non-overlapping CIs).

## bench-server — persistent model for fast iteration

```sh
./target/release/dismantle bench-server \
    --weights models/deepseek-v2-lite-q4.gguf \
    --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
    --stdin
```

Loads the model once; accepts JSON-line requests on stdin, emits JSON-line responses on stdout. Eliminates the 5–15 s model load per iteration. Use `bench_server_driver.sh` for automated multi-request runs.

## bench-kernel — per-kernel micro-benchmarks

```sh
# Single kernel at one shape
./target/release/dismantle bench-kernel \
    --kernel gemv_q4_k_m_v2_pinned_tcb \
    --shape 1408x2048 \
    --iterations 500

# All supported kernels at a shape
./target/release/dismantle bench-kernel --all --shape 1408x2048

# Compare two kernels
bash tools/bench/kernel_compare.sh v2_lite_gate_up \
    gemv_q4_k_m_v2_pinned_tcb gemv_q3_k_pinned_tcb
```

Allocates synthetic buffers (no model load). Times GPU dispatch only. Results append to `bench_results/kernel_perf_history.jsonl`.

## git bisect — finding the regressing commit

See [tools/bisect/README.md](../bisect/README.md). Quick start:

```sh
git bisect start
git bisect bad HEAD
git bisect good <last_known_good>
git bisect run tools/bisect/bisect_v2_lite.sh 17.0
# wait ~30–50 min for 10 commits
git bisect reset
```
