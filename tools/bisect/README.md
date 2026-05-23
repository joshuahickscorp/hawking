# Bisecting a V2-Lite perf regression

Use this when V2-Lite dec_tps drops by ≥5% between two known-good commits
and you need to find which commit introduced the regression.

## Quick start

```bash
# 1. Set the threshold: lower of the two known-good medians, minus ~1 dec_tps
THRESHOLD=18.5

# 2. Start the bisect
git bisect start
git bisect bad HEAD               # current bad state
git bisect good <last_good_commit>

# 3. Run automated bisect (~30-60 min for ~10 commits)
git bisect run tools/bisect/bisect_v2_lite.sh $THRESHOLD

# 4. git bisect reports the first regressing commit. Inspect the diff.
#    Common culprits: Metal shader change, kernel dispatch geometry,
#    quantization block layout, profile json entry for the wrong shader variant.

# 5. End bisect
git bisect reset
```

## Time per step

Each bisect step:
- `cargo build --release --workspace`: ~45s (incremental usually ~10-20s)
- `dismantle autotune`: ~5s
- 4-trial coexist bench: ~3-5 min (includes 30s inter-trial settle)

For N commits between good and bad: `⌈log₂(N)⌉` steps.
- 10 commits → 4 steps → ~15-25 min
- 30 commits → 5 steps → ~20-30 min
- 100 commits → 7 steps → ~30-45 min

## Choosing the threshold

Use the lower of the two confirmed medians, minus ~1 dec_tps slack:
```
known-good median:  19.8 dec_tps
known-bad median:   16.4 dec_tps (after regression)
threshold:          18.5 dec_tps  # = 19.8 - 1.3 (slack for coexist variance)
```

Too tight (< 1 dec_tps slack): false positives from Claude.app GC pauses.
Too loose (> 3 dec_tps slack): misses the exact regressing commit.

If in doubt, use `bench_diff.sh <good> <bad>` first to confirm the regression
is statistically significant before starting a bisect.

## Environment variables

| var | default | description |
|---|---|---|
| `WEIGHTS` | `models/deepseek-v2-lite-q4.gguf` | V2-Lite weights path |
| `BISECT_PROFILE` | `/tmp/dismantle_bisect_profile.json` | per-step autotune output |

Override at bisect start:
```bash
WEIGHTS=/Volumes/fast/v2lite.gguf \
git bisect run tools/bisect/bisect_v2_lite.sh 17.0
```

## How it works

1. `cargo build --release --workspace` at the bisect-selected commit.
2. `dismantle autotune` to regenerate the kernel profile (shader hash may
   differ from main — profile JSON must match the compiled shaders).
3. `TRIALS=4 TOKENS=24 coexist_bench.sh --quiet` — outputs `median: X.X` to
   stdout, which the script greps.
4. Returns 0 (good) if `median >= threshold`, 1 (bad) otherwise.
5. Build failure → exit 1 (bad), because a broken build is a regression.

## After bisect finds the commit

```bash
git show <offending_commit>
```

Common patterns to look for:
- Shader file changed → kernel now uses different code path
- `profiles/*.json` changed → different kernel variant selected
- `kernels/mod.rs` dispatch geometry changed → wrong thread group shape
- Metal buffer offset / alignment change → wrong data read

Use `dismantle bench-kernel --kernel <name> --shape <shape>` to confirm
which specific kernel became slower at a production shape.

## Mixtral variant

For Mixtral 8x7B regressions, the same script works with:
```bash
WEIGHTS=models/mixtral-8x7b-instruct-q4.gguf \
git bisect run tools/bisect/bisect_v2_lite.sh $THRESHOLD
```

(The script uses `coexist_bench.sh` which defaults to V2-Lite. Override
`WEIGHTS` and `PROFILE` env vars, or create a `bisect_mixtral.sh` copy
with different defaults for Mixtral sessions.)
