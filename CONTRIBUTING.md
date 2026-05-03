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
cargo test --workspace --lib
cargo test --release --test phase1_kernel_parity
cargo test --release --test phase2_mla_metal_parity
cargo test --release --test phase2_weight_pinning_parity
cargo test --release --test phase2_moe_block_batched_parity
cargo test --release --test phase2_moe_block_fused_parity
```

The parity regime is tight: max absolute diff < `1e-3` for every
Metal-vs-CPU reference kernel path. Token-output baselines live in
`tests/golden/` and are checked separately via
`tools/haul/token-regression.sh`.

## Code conventions

- `cargo fmt --all` — non-optional. CI rejects unformatted code.
- `cargo clippy --workspace --all-targets` — target ≤ 30 warnings.
- Kernel changes: include a parity test against the CPU reference and a
  `cargo bench` diff in the PR description.
- No clippy `allow` at module level. Per-function `allow` with a comment
  explaining why.

## What's in scope

- Bug fixes
- Performance improvements with a `dismantle bench` measurement
- New MoE architectures (Mixtral, Qwen-MoE, future DeepSeek lines)
- Test coverage
- Documentation improvements

## What's out of scope

- Dense-model support as a primary target. dismantle is a MoE engine.
- Backends beyond Metal (CUDA, ROCm, Vulkan) until v0.3.
- Chat UIs. dismantle is a runtime; bring your own client.
- Trainer code.

## Pull requests

- One concern per PR.
- Every kernel-level change: parity test + benchmark diff.
- Every behavioural change: a test.

## License

By contributing you agree your changes are MIT-licensed (see [LICENSE](./LICENSE)).
