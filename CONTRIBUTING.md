# Contributing to dismantle

## Toolchain

- **macOS** Apple Silicon (M1/M2/M3/M4). M3 Pro 18 GB is the reference for benchmarks.
- **Rust** stable ≥ 1.80.
- **Xcode Command Line Tools** with Metal SDK (any version shipped with macOS 14+).

```sh
rustc --version          # ≥ 1.80
xcrun --show-sdk-version # ≥ 14.0
```

## Build

```sh
cargo build --release --workspace
```

## Correctness gates

Run before any PR. All must pass:

```sh
cargo test --release --workspace --lib
cargo test --release --workspace --tests
```

The parity regime is tight: max absolute diff < `1e-3` for every
Metal-vs-CPU reference kernel path (5e-3 acceptable for fp16 storage,
1e-2 for Q3-class quants). Token-output baselines live in `tests/golden/`.

## Bench gates

Performance changes require a measured bench delta:

```sh
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh    # before
# ... apply your change ...
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh    # after
bash tools/bench/bench_diff.sh HEAD~1 HEAD              # significance
```

`bench_diff.sh` reports whether the change is statistically significant
given trial-to-trial variance. Defaults flip in profile JSON only if
the change clears the +5% gate.

For per-kernel claims:

```sh
./target/release/dismantle bench-kernel --kernel <name> --shape <RxC> --iterations 200
```

## Code conventions

- `cargo fmt --all` — non-optional.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- Kernel changes: include a parity test against the CPU reference + a
  bench-kernel timing in the PR description.
- No clippy `allow` at module level. Per-function `allow` with a comment
  explaining why.
- Internal accumulators (variance, softmax, MAC) stay f32 even when
  storage is f16. Non-negotiable for numerical stability.

## What's in scope

- Bug fixes
- Performance improvements with a measured bench delta
- New MoE architectures (Mixtral, Qwen-MoE, future DeepSeek lines)
- Test coverage
- Documentation improvements
- Apple Neural Engine, AMX, MPSGraph integration (long-term perf work)

## What's out of scope (for now)

- Dense-model support as a primary target. dismantle is a MoE engine.
- Non-Apple backends (CUDA, ROCm, Vulkan).
- Chat UIs. dismantle is a runtime; bring your own client.
- Trainer code.

## Pull requests

- One concern per PR.
- Kernel-level change: parity test + bench-kernel measurement in description.
- Behavioural change: a test.
- New optional feature: bench-first commit gate (must clear +5% e2e on
  V2-Lite to flip default; else lands as opt-in only).

## License

By contributing you agree your changes are MIT-licensed (see [LICENSE](./LICENSE)).
