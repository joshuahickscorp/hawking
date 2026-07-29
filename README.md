# hawking

`hawking` runs quantized language models on Apple Silicon GPUs. It is one Rust binary that
loads a GGUF model file, runs it through hand-written Metal kernels, and serves it over an
OpenAI-compatible HTTP API.

Everything is built in-tree: no Python runtime, no `llama.cpp`, no BLAS, no MPSGraph. The
point is an inference stack you can read, test, and change end to end.

## Status

- Best-tested model: Qwen2.5-3B-Instruct in Q4_K_M.
- Measured baseline: about 31 decode tok/s on Qwen2.5-3B-Q4_K_M, on an M3 Pro with 18 GB,
  in clean-room runs. That number is deliberately modest and it is measured, not projected.
- Other model families run on the same code, but how much each one has been verified
  varies. Check [MODELS.md](MODELS.md) before you rely on one.
- Active development. Expect sharp edges.

## What it does

- Loads GGUF weights with a single mmap and no copies.
- Runs Q4_K / Q6_K matrix-vector kernels, attention (standard and MLA), RoPE, RMSNorm, and
  sampling on the GPU. The Metal source is embedded and compiled at runtime.
- Handles these architectures today: Qwen2.5 dense, Llama, Qwen3-MoE, DeepSeek-V2, RWKV-7.
- Serves `/v1/chat/completions` and `/v1/completions` with streaming, plus `/healthz` and a
  Prometheus `/metrics` endpoint. Requests are batched continuously.
- Checks whether a model fits in your Mac's memory before loading it (`doctor`, `fit`), and
  times kernel variants on your own machine to pick the fastest (`autotune`).
- Has a pure-Rust CPU path, used to check the Metal kernels produce the same tokens and to
  build off macOS.
- Ships a desktop front-end, HIDE, under `app/`: a React/TypeScript chat and code interface
  that talks to the same engine, plus a Tauri v2 shell that supervises the local server. The
  web front-end is the developed part; the native shell is a build-ready scaffold that has not
  been compiled yet.

## Build

You need an Apple Silicon Mac, Rust 1.80 or newer, and the Xcode command line tools.
Qwen2.5-3B in Q4_K_M needs about 4 GB of RAM.

```sh
git clone https://github.com/joshuahickscorp/hawking.git
cd hawking
cargo build --release --workspace
```

The binary lands at `target/release/hawking`. [ARCHITECTURE.md](ARCHITECTURE.md) is the
internal map if you want to change something.

## Get a model

Put any GGUF file in `models/` and pass it with `--weights`. The best-tested target is
Qwen2.5-3B-Instruct-Q4_K_M, which you download yourself. There is also
`./tools/fetch-model.sh`, but note it fetches DeepSeek-V2-Lite-Chat Q4_K_M, not the Qwen
model used in the examples below.

## Usage

```sh
M=models/qwen2.5-3b-instruct-q4_k_m.gguf

# Will this model fit on this Mac?
hawking doctor --weights $M

# Time the kernels on this machine and save the winners.
hawking autotune --weights $M --out profiles/my-mac.json

# Generate text.
hawking generate --weights $M --kernel-profile profiles/my-mac.json \
  --prompt "Write a Rust function that reverses a linked list." --max-new-tokens 256

# Serve an OpenAI-compatible API.
hawking serve --weights $M --kernel-profile profiles/my-mac.json --addr 127.0.0.1:8080
```

`profiles/` already ships tuned profiles for an M3 Pro 18 GB (Qwen2.5 0.5B/1.5B/3B/7B and
DeepSeek-V2-Lite), so you can skip `autotune` on that machine. [docs/serve.md](docs/serve.md)
covers the API; `hawking --help` lists the rest (`bench`, `tokenize`, `verify`, `press`, and
others). One of those, `hawking press --dry-run`, prints a plan for compressing a model too
large to hold in memory. It only prints the plan — it cannot produce a compressed model yet.

## Performance

Benchmark numbers are only trustworthy on an otherwise idle GPU; a heavy background app can
inflate them several times over. The harness, its caveats, and the head-to-head run against
llama.cpp and MLX are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Optimizations that were
tried and rejected are written up in [docs/dead_levers.md](docs/dead_levers.md).

```sh
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh
bash tools/bench/clean_room_batch.sh
```

## What's next

The near-term work is stabilization: finishing wiring that already exists, adding gates for
model families that don't have them, and cleaning up docs before a release. See
[ROADMAP.md](ROADMAP.md).

## Maintainer and license

Joshua Hicks. MIT — see [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).
