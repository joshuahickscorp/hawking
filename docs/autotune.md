# Autotune

dismantle ships a kernel-selection autotuner that benchmarks candidate
dispatch strategies on your specific hardware and locks the winner into a
profile JSON. Subsequent `generate`, `serve`, and `bench` runs load that
profile and skip re-benchmarking.

## Running autotune

```sh
dismantle autotune \
  --weights models/deepseek-v2-lite-q4.gguf \
  --profile m3-pro-18gb \
  --max-hours 8 \
  --out profiles/deepseek-v2-lite-q4.m3pro18.json
```

The profile name (`m3-pro-18gb`) is a free-form label embedded in the output
file. `--max-hours` caps the total wall time; autotune stops early if all
candidates have been measured.

The run also writes `profiles/deepseek-v2-lite-q4.m3pro18.jsonl` — a
line-delimited candidate log suitable for diffing across runs or machines.

## Using a profile

Pass `--kernel-profile` to any subcommand that dispatches kernels:

```sh
dismantle generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --prompt "Hello" \
  --max-new-tokens 16

dismantle bench \
  --backend dismantle \
  --suite decode \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --trials 3 \
  --max-new-tokens 64 \
  --json /tmp/bench.json
```

Or via environment variable:

```sh
export DISMANTLE_KERNEL_PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json
dismantle generate --weights models/deepseek-v2-lite-q4.gguf --prompt "Hello" --max-new-tokens 16
```

## When to re-run autotune

- After a `cargo build --release` that changes any Metal shader (the profile
  embeds a shader hash; a mismatch causes a safe fallback, not a crash).
- After moving to a different Mac (device name and GPU tier are part of the
  profile identity).
- After a major OS/driver update that affects Metal performance.

## Checking the shader hash

```sh
dismantle shader-hash
```

Compare this against the `shader_hash` field in your profile JSON. A mismatch
means the profile was generated from a different shader revision; re-run
autotune or delete the stale profile and let the runtime fall back to the
deterministic default.

## Speculative decode token regression

The 50-prompt token baseline can run through a profiled path:

```sh
DISMANTLE_KERNEL_PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json \
DISMANTLE_SPECULATE=exact-shared \
DISMANTLE_VERIFY_WINDOW=4 \
tools/haul/token-regression.sh tests/golden/_phase2_token_baseline_50.hashes
```
