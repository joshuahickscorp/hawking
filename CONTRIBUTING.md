# Contributing to dismantle

Thanks for your interest. dismantle is in **Phase 0** as of this
writing — not yet runnable, no v0.1.0 release. Outside contributions
are welcome, but the core architectural decisions and the v0.1
roadmap are not open for revision until after launch.

## Toolchain

- **macOS** (Apple Silicon — M1, M2, M3, M4 all supported in
  principle; M3 Pro 18 GB is the reference target for Phase 0–5
  benchmarks).
- **Rust** ≥ 1.80 (stable channel).
- **Xcode Command Line Tools** with the Metal SDK (any version
  shipped with macOS 14+).
- **Optional:** a `llama.cpp` build for the `wax-vs-llama-cpp`
  benchmark suite. The pinned commit hash is recorded in
  `docs/m3_audit.md`.

```sh
# One-time toolchain check:
rustc --version            # ≥ 1.80
xcrun --show-sdk-version   # ≥ 14.0
```

## Build

```sh
cargo build --release
./target/release/dismantle version
```

## Test

```sh
cargo test --workspace
```

Once Phase 0 lands, `cargo test -p dismantle-core --test correctness`
runs the 50-prompt equivalence suite against a frozen `llama-cli`
reference.

## Benchmark

```sh
cargo run --release -p dismantle -- bench \
    --weights ./deepseek-v2-lite-q4.gguf \
    --model deepseek-v2-lite-q4 \
    --suite all
```

The headline `wax-vs-llama-cpp` suite needs a sibling `llama-cli`
binary; see `docs/m3_audit.md` for the pinned upstream commit.

## What's in scope

- Bug fixes
- Performance improvements (with a `dismantle bench` measurement)
- New MoE architectures (Mixtral, future Qwen-MoE / DeepSeek lines)
- Test coverage
- Documentation improvements

## What's out of scope (please don't open a PR)

- Dense-model support. dismantle is a MoE engine; use llama.cpp.
- Backends beyond Metal (CUDA, ROCm, Vulkan) until v0.3.
- Chat UIs. dismantle is a runtime.
- Trainer code.

## Code conventions

- `cargo fmt` is non-optional. CI rejects unformatted code.
- `cargo clippy --workspace -- -D warnings` is non-optional.
- Public APIs in `dismantle-core` get rustdoc on the type and on
  every public method. Internal modules get a one-paragraph
  module-level doc.
- `.metal` shaders get a header comment block listing every
  kernel, its phase, and its wedge (if applicable).

## Pull requests

- One concern per PR. Refactors and feature work do not mix.
- Every kernel-level change includes a benchmark output diff in the
  PR description.
- Every behavioural change includes a test.

## License

By contributing you agree your changes are MIT-licensed (see
[LICENSE](./LICENSE)).
