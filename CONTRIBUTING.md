# Contributing

`dismantle` is owner-maintained. Public contributor credit in the project docs is
kept to the maintainer:

- Joshua Hicks

Outside fixes and reports are still welcome, but changes should stay small,
measured, and easy to review.

## Toolchain

- Apple Silicon Mac for Metal work
- Rust stable 1.80 or newer
- Xcode Command Line Tools with Metal SDK

```sh
rustc --version
xcrun --show-sdk-version
cargo build --release --workspace
```

## Correctness

Run the core gates before opening a PR:

```sh
cargo fmt --all
cargo test --release --workspace --lib
cargo test --release --workspace --tests
```

Kernel changes need a parity test against the CPU reference path. Behavioral
changes need a focused test.

## Performance

Performance claims need before/after numbers:

```sh
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh
bash tools/bench/bench_diff.sh HEAD~1 HEAD
```

Defaults should only change when the win is measured and repeatable.

## Pull Requests

- Keep one concern per PR.
- Keep new features default-off until correctness and performance are proven.
- Include the command output that supports the change.
- Do not add broad `allow` attributes for clippy warnings.
- Keep documentation honest about what is verified versus experimental.

## License

By contributing you agree your changes are MIT-licensed. See [LICENSE](LICENSE).
