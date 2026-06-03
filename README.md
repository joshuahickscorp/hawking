# dismantle

Pure-Rust inference engine for transformer language models on Apple Silicon — single binary, no Python at runtime, no llama.cpp dependency.

## What it does

- Loads GGUF weights via mmap with a zero-copy `MTLBuffer` over the mapping (no second allocation).
- Runs dense and Mixture-of-Experts transformers through hand-rolled Metal compute kernels.
- Exposes an OpenAI-compatible HTTP API (`dismantle serve`) and a benchmark harness (`dismantle bench`).
- Architecture is auto-detected from GGUF metadata; unknown architectures error with the supported list.

## Supported families

| Family | Kind |
|---|---|
| Qwen2.5 (`qwen2` / `qwen2.5`) | dense — primary tuned target |
| Llama 3.x / Mistral (`llama3.x`, `mistral`) | dense |
| Gemma 2 (`gemma2`) | dense |
| Phi-3 / 3.5 (`phi3`) | dense |
| DeepSeek-V2-Lite (`deepseek2-lite`) | MoE — 16B params, 2.4B active/token, MLA attention |
| Mixtral 8×7B (`llama`+MoE) | MoE |
| Qwen3-MoE (`qwen3moe`) | MoE |

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

Or place any GGUF in `models/` and pass it via `--weights`.

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

## Performance (M3 Pro 18 GB, clean-room, 2026-05-31)

| model | quant | dec_tps | notes |
|---|---|---:|---|
| Qwen2.5-3B-Instruct | Q4_K_M | ~31 | clean-room anchor (Claude.app quit), greedy temp=0 |
| DeepSeek-V2-Lite-Chat | Q4_K_M | ~17 | TRIALS=4 TOKENS=24, 95% CI [16.6, 18.0] |
| Mixtral-8×7B-Instruct | Q3_K_M | ~0.1 | SSD-bandwidth-limited on 18 GB |

llama.cpp Metal reaches ~50 dec_tps on Qwen-3B-Q4_K_M. dismantle's ~31 is the bandwidth-bound ceiling for batch-1 Q4_K GEMV on this GPU; further gains require fewer weight bytes or the speculative/stateful axes.

## Moat

Three levers that survive clean measurement:

1. **Prefix-cache reuse** — default-on exact KV prefix cache; elides up to ~84% of prefill on warm shared-prefix workloads. On-disk variant persists across runs.
2. **Speculative decode on code** — free n-gram draft (τ=1.43 on code) with pruned-vocab GPU verify; +148% decode on repetitive code, bit-identical to greedy.
3. **Low-RAM footprint** — zero-copy loader keeps peak RSS near model size; a 3B runs alongside other GPU-heavy work on an 18 GB machine.

The rest is recorded in `reports/dead_levers.md`.

## Named profiles — `--profile fast` / `--profile deterministic`

Default decode uses **f16-scales predec** (a measured +9.3% tps / −1.4% J/tok both-axes win; quality-equivalent, **not** bit-identical to the pre-2026-06-03 default — f16 scale rounding perturbs logits ~5e-4 relative). `--profile deterministic` (or `DISMANTLE_QWEN_PREDEC_F32SCALES=1`) restores the bit-identical f32-scales path.

`--profile fast` additionally sets `DISMANTLE_QWEN_VOCAB_PRUNE=32000` + Q4K LM-head + Q4K FFN-down on top of the default predec + f16-scales. Explicit `DISMANTLE_QWEN_*` env vars take precedence over the bundle.

## CPU reference path — `DISMANTLE_FORCE_CPU=1`

Forces the engine to load with no Metal context, exercising the same pure-Rust path used off-macOS. Cross-checked on Qwen2.5-0.5B-Q4_K_M: 12/12 leading greedy token IDs identical vs Metal. Dense models only; MoE CPU decode is out of scope.

## Reproduce the perf numbers

```sh
# Paired median + 95% CI (Claude.app can be running)
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh

# Contamination-controlled absolute anchor (quit Claude.app first)
bash tools/bench/clean_room_batch.sh
```

See [tools/bench/README.md](tools/bench/README.md) for parameter conventions.

## Notes

- Mixtral Q3_K_M is SSD-bandwidth-limited on 18 GB; see [docs/mixtral.md](docs/mixtral.md).
- Off-macOS: Metal deps are macOS-gated in `Cargo.toml`; CPU primitives compile unconditionally. Off-macOS verification requires a non-macOS toolchain.
- v0.2.x, active development. See [ARCHITECTURE.md](ARCHITECTURE.md) for the internal map.

## License

MIT. See [LICENSE](LICENSE).
