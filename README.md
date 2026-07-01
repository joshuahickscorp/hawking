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
- `generate`, `serve`, `bench`, `doctor`, `autotune`, `fit`, and `press` CLI workflows.
- Apple Fit: `hawking fit` and `doctor --json` predict the strongest usable
  configuration for the current Mac; `serve --auto --intent <…>` selects it
  (capability-first — it reports the full envelope and never hides a throttle).
- Condense (low-bit model press): `hawking press --dry-run --memory-budget` plans
  out-of-core artifact creation for parent models too large to hold fully resident.
- CPU reference path for off-macOS builds and Metal parity checks.
- Prefix-cache reuse, speculative decode experiments, and benchmark tooling.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal map and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the SOTA comparison harness
(Hawking vs llama.cpp vs MLX).

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

## Roadmap

Hawking is a from-scratch LLM inference engine for Apple Silicon, written in
Rust with hand-written Metal kernels. It runs quantized GGUF models end to end
on the GPU, with no PyTorch, llama.cpp, or BLAS. The work ahead is about
pushing quality, memory, and context length further on Apple hardware, with the
heavy runs on an Apple M2 Max Mac Studio (96 GB).

### Now (works today)
- Dense Qwen2.5 forward pass on Metal, GGUF-native, with Q4_K / Q6_K kernels,
  RoPE / RMSNorm / attention / GPU sampling, and an OpenAI-compatible server.
- CPU-to-GPU numerical-parity tests and golden-hash regression gates running in
  CI on real Apple Silicon.

### Next
- RWKV-7 (SSM) long-context path: flat-cost decode with no KV-cache wall.
- Per-channel int4 KV cache, to cut KV memory by roughly three quarters.
- Post-hoc context extension (YaRN RoPE-scaling) validated by needle-in-a-haystack
  retrieval, not just "it didn't crash" - stretch the trained window at serve time,
  paired with int4 KV so the longer context actually fits in memory.
- STKV, a tiered KV hybrid that is Hawking-specific because it uses both engines at
  once: exact int8 recall for attention sinks and the recent window, a trellis-coded
  warm band (the same codec as the weights, on the cache), and an unbounded cold tail
  that is either paged to SSD (lossless, slow) or summarized into an RWKV-7 state
  (lossy, flat memory). Exact recall where attention lands, unbounded reach beyond it.
- Close the remaining decode-throughput gap to llama.cpp / MLX (kernel and
  scheduling work).

### Later
- Condense: an out-of-core, memory-budgeted low-bit compression pipeline that
  can quantize models too large to hold resident, so a single Mac can prepare
  and serve models well beyond its own memory.
- The size frontier: stop requiring the whole model resident. The parameter
  ceiling is then storage, not RAM - the model lives on the SSD and only the
  weights a token touches stream through memory. For MoE (all the giant models),
  only the routed experts page in, so a multi-trillion-parameter model runs at a
  usable rate where a RAM-resident engine tops out near half a trillion. An auto
  advisor picks the bit format and the serve regime (resident / expert-paged /
  dense out-of-core) per model and device.
- Broader verified architecture coverage (MoE, Mamba2, more dense families)
  under the same correctness-before-speed gates.
