# hawking

`hawking` is a from-scratch LLM inference engine for Apple Silicon. It loads
GGUF models with mmap, runs hand-written Metal kernels from Rust, and exposes a
small CLI plus an OpenAI-compatible HTTP server.

It is a systems project first: no Python runtime, no `llama.cpp` dependency, no
BLAS, and no MPSGraph. The goal is an auditable inference stack that can be
measured, tested, and changed without hiding work in external runtimes.

## Status

- Primary tuned target: Qwen2.5 dense GGUF, especially Q4_K_M.
- Dense and MoE families share the same runtime, but verification varies by
  model. Check [MODELS.md](MODELS.md) before relying on a family.
- Current clean-room baseline on an M3 Pro 18 GB: about 31 decode tok/s on
  Qwen2.5-3B-Q4_K_M.
- Active development. Expect sharp edges.

## Features

- Zero-copy GGUF weight loading through mmap-backed Metal buffers.
- Hand-written Metal kernels for Q4_K / Q6_K GEMV, attention, RoPE, RMSNorm,
  sampling, and fused paths.
- OpenAI-compatible `/v1/chat/completions` and `/v1/completions` endpoints.
- `generate`, `serve`, `bench`, `doctor`, and `autotune` CLI workflows.
- CPU reference path for off-macOS builds and Metal parity checks.
- Prefix-cache reuse, speculative decode experiments, and benchmark tooling.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal map.

## Build

Requirements:

- Apple Silicon Mac for the Metal path
- Rust stable 1.80 or newer
- Xcode Command Line Tools
- About 4 GB RAM for Qwen2.5-3B Q4_K_M

```sh
git clone https://github.com/joshuahickscorp/hawking.git
cd hawking
cargo build --release --workspace
```

The binary is written to `target/release/hawking`.

## Get A Model

```sh
./tools/fetch-model.sh
./tools/fetch-mixtral.sh
```

You can also place any GGUF file in `models/` and pass it with `--weights`.
The best-tested target is Qwen2.5-3B-Instruct-Q4_K_M.

## Usage

```sh
# Check whether the model fits before loading it.
hawking doctor --weights models/qwen2.5-3b-instruct-q4_k_m.gguf

# Tune kernels for this machine.
hawking autotune \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --out profiles/my-mac.json

# Generate text.
hawking generate \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --kernel-profile profiles/my-mac.json \
  --prompt "Write a Rust function that reverses a linked list." \
  --max-new-tokens 256

# Serve an OpenAI-compatible API.
hawking serve \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --kernel-profile profiles/my-mac.json \
  --addr 127.0.0.1:8080
```

See [docs/serve.md](docs/serve.md) for API details.

## Performance

The headline number is intentionally modest and measured: Qwen2.5-3B-Q4_K_M
runs at about 31 decode tok/s on an M3 Pro 18 GB in clean-room runs. The project
keeps a kill-ledger of optimizations that were tested and rejected in
[docs/dead_levers.md](docs/dead_levers.md).

Useful bench entry points:

```sh
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh
bash tools/bench/clean_room_batch.sh
```

See [tools/bench/README.md](tools/bench/README.md) for conventions.

## Contributors

- Joshua Hicks

## License

MIT. See [LICENSE](LICENSE).
