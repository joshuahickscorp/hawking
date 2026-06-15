# dismantle

A from-scratch LLM inference engine for Apple Silicon — a single pure-Rust binary that mmaps a GGUF model, decodes it through hand-written Metal compute kernels, and serves it over an OpenAI-compatible API. No Python at runtime, no `llama.cpp` dependency, no BLAS or MPSGraph.

## What this is — and what it isn't

A **systems project**. The goal is to own the entire inference stack end-to-end — GGUF parsing → zero-copy weight residency → Metal Q4_K GEMV / attention / RoPE / RMSNorm kernels → sampling → an OpenAI-compatible HTTP server — in one auditable Rust binary.

It is **not** a `llama.cpp` replacement. llama.cpp is faster and supports far more models and quant formats (≈50 vs ≈31 decode tok/s on Qwen2.5-3B-Q4_K_M, M3 Pro). dismantle exists to build that stack from scratch and hold it to a strict correctness bar — not to win on model coverage or peak throughput.

Where the actual work is:

- **Hand-written Metal kernels** — Q4_K / Q6_K GEMV, flash-style attention, RoPE, RMSNorm, and fused variants, with no BLAS, MPSGraph, or external kernel library.
- **Bit-parity discipline** — GPU kernels are gated against a CPU reference port (and, increasingly, an independent `llama.cpp` logit oracle), so a refactor can't silently change model outputs.
- **A measured kill-ledger** — [`docs/dead_levers.md`](docs/dead_levers.md) records the optimizations that were tried and **rejected**, each with the measurement that killed it. The honest negative results are the most useful artifact here.

[ARCHITECTURE.md](ARCHITECTURE.md) is the internal map.

## What it does

- Loads GGUF weights via mmap with a zero-copy `MTLBuffer` over the mapping (no second allocation).
- Runs dense and Mixture-of-Experts transformers through hand-rolled Metal compute kernels.
- Exposes an OpenAI-compatible HTTP API (`dismantle serve` — both `/v1/chat/completions` and `/v1/completions`) and a benchmark harness (`dismantle bench`).
- Auto-detects the architecture from GGUF metadata; unknown architectures error with the supported list.

## Models

Architecture is detected from GGUF metadata. The primary tuned and verified target is **Qwen2.5 dense (Q4_K_M)**. Other families load through the same path but are verified to varying degrees — check [MODELS.md](MODELS.md) for the exact *verified / loads / untested* status per family before relying on one.

| Family | Kind |
|---|---|
| Qwen2.5 (`qwen2`) | dense — primary tuned target |
| Llama 3.x / Mistral | dense |
| Gemma 2 | dense |
| Phi-3 / 3.5 | dense |
| DeepSeek-V2-Lite | MoE — MLA attention |
| Mixtral 8×7B | MoE |
| Qwen3-MoE | MoE |

## Build

```sh
git clone https://github.com/joshuahickscorp/dismantle.git
cd dismantle
cargo build --release --workspace
# binary: target/release/dismantle
```

Requirements: Apple Silicon Mac (M1–M4), Rust stable, ~4 GB RAM for Qwen2.5-3B Q4_K_M.

## Get a model

```sh
./tools/fetch-model.sh        # DeepSeek-V2-Lite Q4_K_M (~9.7 GB)
./tools/fetch-mixtral.sh      # Mixtral 8×7B Q3_K_M (~16 GB)
```

Or place any GGUF in `models/` and pass it via `--weights`. The tuned target is Qwen2.5-3B-Instruct-Q4_K_M.

## Usage

```sh
# Check fit before loading
dismantle doctor --weights models/qwen2.5-3b-instruct-q4_k_m.gguf

# Pick fastest kernels for your machine (run once, ~1–2 min)
dismantle autotune \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --out profiles/my-mac.json

# Generate
dismantle generate \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --kernel-profile profiles/my-mac.json \
  --prompt "Write a Rust function that reverses a linked list." \
  --max-new-tokens 256

# Serve (OpenAI-compatible)
dismantle serve \
  --weights models/qwen2.5-3b-instruct-q4_k_m.gguf \
  --kernel-profile profiles/my-mac.json \
  --addr 127.0.0.1:8080
```

`/v1/chat/completions` and `/v1/completions` both stream via SSE. See [docs/serve.md](docs/serve.md).

## Performance (M3 Pro 18 GB, clean-room)

llama.cpp Metal reaches ≈50 decode tok/s on Qwen2.5-3B-Q4_K_M; dismantle reaches ≈31. That gap is the honest baseline: ≈31 is near the bandwidth-bound ceiling for batch-1 Q4_K GEMV on this GPU, and closing it further needs *fewer weight bytes* (sub-4-bit quant) or the speculative / stateful axes — not micro-optimization. The measurements that ruled out the micro-optimization paths are in [`docs/dead_levers.md`](docs/dead_levers.md).

| model | quant | dec_tps | notes |
|---|---|---:|---|
| Qwen2.5-3B-Instruct | Q4_K_M | ~31 | clean-room anchor, greedy temp=0 |
| DeepSeek-V2-Lite-Chat | Q4_K_M | ~17 | TRIALS=4 TOKENS=24, 95% CI [16.6, 18.0] |
| Mixtral-8×7B-Instruct | Q3_K_M | ~0.1 | SSD-bandwidth-limited on 18 GB |

## Engineering depth — what survives clean measurement

Three levers held up under contamination-controlled measurement (the rest are in the kill-ledger):

1. **Prefix-cache reuse** — default-on exact KV prefix cache; elides up to ~84% of prefill on warm shared-prefix workloads. An on-disk variant persists across runs.
2. **Speculative decode on code** — free n-gram draft (τ=1.43 on code) with pruned-vocab GPU verify; +148% decode on repetitive code, bit-identical to greedy.
3. **Low-RAM footprint** — the zero-copy loader keeps peak RSS near model size, so a 3B runs alongside other GPU-heavy work on an 18 GB machine.

## Known limitations

- **Single-stream is the tuned path.** Continuous batching (the multi-sequence decode lane) is implemented and parity-gated bit-identical to single-stream, but it is less battle-tested and the batch scheduler does not yet honor `stop` strings.
- **Model coverage is narrow** compared to llama.cpp — see [MODELS.md](MODELS.md) for what's actually verified.
- **Off-macOS** builds compile the CPU path only (dense models); MoE CPU decode is out of scope.
- **Greedy decoding can repeat.** `generate` defaults to `--temperature 0` (deterministic greedy), which can fall into short repetition loops on long outputs. Pass `--temperature 0.7` (or raise `--top-p`) for varied text; the chat/completions server already defaults to temperature 0.7.

## Named profiles — `--profile fast`

`--profile fast` sets `DISMANTLE_QWEN_VOCAB_PRUNE=32000` + Q4K LM-head + Q4K FFN-down + predec + f16-scales. Delivers +7.4% paired dec_tps on Qwen2.5-3B-Q4_K_M; output is **not** bit-identical to default (f16 scale rounding). Explicit `DISMANTLE_QWEN_*` env vars take precedence over the bundle.

## CPU reference path — `DISMANTLE_FORCE_CPU=1`

Forces the engine to load with no Metal context, exercising the same pure-Rust path used off-macOS. Cross-checked on Qwen2.5-0.5B-Q4_K_M: 12/12 leading greedy token IDs identical vs Metal. Dense models only; MoE CPU decode is out of scope.

## Reproduce the perf numbers

```sh
# Paired median + 95% CI (shared background load is OK)
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh

# Contamination-controlled absolute anchor (close agent/GPU workloads first)
bash tools/bench/clean_room_batch.sh
```

See [tools/bench/README.md](tools/bench/README.md) for parameter conventions.

## Notes

- Mixtral Q3_K_M is SSD-bandwidth-limited on 18 GB; see [docs/mixtral.md](docs/mixtral.md).
- Off-macOS: Metal deps are macOS-gated in `Cargo.toml`; CPU primitives compile unconditionally. Off-macOS verification requires a non-macOS toolchain.
- v0.2.x, active development. See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal map.

## Roadmap

- **Sub-4-bit quantization (deferred).** Absorbing the `strand-quant` QTIP trellis backend to bake `.strand` artifacts with integer, float-free decode is planned but **not implemented** — `tools/strand_bake` is a labeled placeholder, not a working baker. dismantle currently consumes pre-quantized GGUF only.
- **Independent-oracle parity in CI** — expand the `llama.cpp` logit-export gate beyond the current self-consistency (Metal vs the in-repo CPU port) checks.

## License

MIT. See [LICENSE](LICENSE).
