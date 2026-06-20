# tools/bisect/

Automated perf-regression bisect for dismantle. Use when dec_tps drops ≥5% between two known-good commits.

## Quick start

```sh
# 1. Set threshold: lower of the two known-good medians, minus ~1 dec_tps slack
THRESHOLD=18.5

# 2. Start
git bisect start
git bisect bad HEAD
git bisect good <last_good_commit>

# 3. Run automated bisect (~30–60 min for ~10 commits)
git bisect run tools/bisect/bisect_v2_lite.sh $THRESHOLD

# 4. Inspect the first regressing commit
git show <offending_commit>

# 5. End
git bisect reset
```

## Time per step

Each step: `cargo build --release` (~10–45 s incremental) + `dismantle autotune` (~5 s) + 4-trial coexist bench (~3–5 min).

| commits in range | steps | total time |
|---|---|---|
| 10 | 4 | ~15–25 min |
| 30 | 5 | ~20–30 min |
| 100 | 7 | ~30–45 min |

## Choosing the threshold

```
known-good median:  19.8 dec_tps
known-bad median:   16.4 dec_tps
threshold:          18.5 dec_tps  # lower good − ~1 dec_tps slack
```

Too tight (< 1 dec_tps slack): false positives from GPU GC pauses.
Too loose (> 3 dec_tps slack): misses the exact regressing commit.

Confirm the regression is statistically significant with `bench_diff.sh <good> <bad>` before starting.

## Environment variables

| var | default | description |
|---|---|---|
| `WEIGHTS` | `models/deepseek-v2-lite-q4.gguf` | weights path |
| `BISECT_PROFILE` | `/tmp/dismantle_bisect_profile.json` | per-step autotune output |

## How it works

1. `cargo build --release --workspace` at the bisect-selected commit.
2. `dismantle autotune` to regenerate the kernel profile (shader hash may differ from main).
3. `TRIALS=4 TOKENS=24 coexist_bench.sh --quiet` — greps `median: X.X` from stdout.
4. Returns 0 (good) if `median >= threshold`, 1 (bad) otherwise. Build failure → 1.

## After bisect finds the commit

Common culprits:
- Shader file changed → kernel uses a different code path.
- `profiles/*.json` changed → different kernel variant selected.
- `kernels/mod.rs` dispatch geometry changed → wrong thread-group shape.
- Metal buffer offset / alignment change → wrong data read.

Confirm with `dismantle bench-kernel --kernel <name> --shape <shape>`.

## Mixtral variant

```sh
WEIGHTS=models/mixtral-8x7b-instruct-q4.gguf \
git bisect run tools/bisect/bisect_v2_lite.sh $THRESHOLD
```
