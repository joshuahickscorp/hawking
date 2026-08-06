# hawking

Runs quantized language models on Apple Silicon GPUs. One Rust binary: it mmaps a GGUF file, pushes
it through hand-written Metal kernels, and serves it over an OpenAI-compatible HTTP API. No Python,
no llama.cpp, no BLAS, no MPSGraph: the whole stack is in-tree and readable.

## How it works

Weights are mmapped once with no copies. Q4_K and Q6_K matrix-vector kernels, attention (standard
and MLA), RoPE, RMSNorm and sampling run on the GPU; the Metal source is embedded and compiled at
runtime. A pure-Rust CPU path checks the kernels emit the same tokens, and builds off macOS.
Architectures today: Qwen2.5 dense, Llama, Qwen3-MoE, DeepSeek-V2, RWKV-7.

`serve` gives `/v1/chat/completions` and `/v1/completions` with streaming, plus `/healthz` and
Prometheus `/metrics`, batching continuously. `doctor` and `fit` say whether a model fits this Mac
before loading; `autotune` picks the fastest kernel variants here; `press --dry-run` plans a
compression for a model too big to hold in memory but cannot bake one yet. `app/` is HIDE, a
React/TypeScript front-end on the same engine: the web side is developed, the Tauri v2 shell a
scaffold never compiled.

## Build and run

Apple Silicon, Rust 1.80 or newer, Xcode command line tools. Qwen2.5-3B Q4_K_M needs about 4 GB.
Bring your own GGUF. `profiles/` ships tuned profiles for an M3 Pro 18 GB.
[ARCHITECTURE.md](ARCHITECTURE.md) maps the internals, [docs/serve.md](docs/serve.md) the API.

```sh
cargo build --release --workspace          # binary lands at target/release/hawking
M=models/qwen2.5-3b-instruct-q4_k_m.gguf
hawking doctor --weights $M
hawking serve  --weights $M --kernel-profile profiles/qwen3b-instruct-q4k.m3pro18.json
```

## Numbers

About 31 decode tok/s on Qwen2.5-3B-Q4_K_M, M3 Pro 18 GB, clean-room greedy decode. Measured, not
projected, and the lower of two unreconciled clean-room anchors. Absolute numbers mean nothing off
an idle GPU: a heavy background app inflates them several times over. Bench harness, caveats and
rejected optimizations: [docs/BENCHMARKS.md](docs/BENCHMARKS.md),
[docs/dead_levers.md](docs/dead_levers.md). Only Qwen2.5-3B and 0.5B have parity or quality gates;
every other family runs ungated or untested, so read [MODELS.md](MODELS.md) first.

## What's next

Stabilization: finishing wiring that exists, gating the ungated families, pulling docs back in line
with the code. Full list in [ROADMAP.md](ROADMAP.md). Two research tracks also live here and neither
has trained anything real. `odyssey/` is a prepared continued-training campaign against a 92 GB
compressed artifact: the trainer is proven on numpy toy fixtures only, the launch fence is false,
the corpora were never collected, the substrate is refused. `ramanujan/` is the math-research half,
a governance layer of roles, an append-only ledger, memory stores, tribunal and graveyard, tested on
fixtures under non-production authority; five small components trained there on CPU, of which
retriever and value beat their baselines while formalizer, prover and repair did not beat the
majority class. It splits into its own repo once a frozen math model exists to direct it.

MIT. Joshua Hicks.
