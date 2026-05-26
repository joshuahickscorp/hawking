# dismantle

Pure-Rust inference engine for Mixture-of-Experts language models on Apple Silicon. Single binary. No Python at runtime. No llama.cpp dependency. Loads GGUF weights via mmap and runs them through hand-rolled Metal compute kernels.

Currently supports:
- **DeepSeek-V2-Lite Q4_K_M** (16B params, 2.4B active per token) — primary tuning target
- **Mixtral 8×7B Q3_K_M** (~16 GB) — runs on 18 GB Macs via memory-conscious expert dispatch
- Generic GGUF loading for Llama / Qwen / DeepSeek architectures

## What's distinctive

- **Pure Rust + Metal** — single binary, no Python in the runtime, no C++ shim. Source-build with `cargo`.
- **MoE-first architecture** — built around expert routing semantics from the kernel level up rather than retrofitting MoE onto a dense engine.
- **Open methodology** — every perf claim in this README is reproducible with the `dismantle bench-server` + `dismantle bench-kernel` tooling included in tree. Statistical CIs, kernel-level timing, cross-commit diffing all built in.
- **Reproducible kernel autotune** — `dismantle autotune` deterministically picks kernel variants for your specific GPU.
- **OpenAI-compatible HTTP API** — `dismantle serve` exposes `/v1/chat/completions`.

## Measured performance (M3 Pro 18 GB, May 2026)

| model | quant | dec_tps (default) | notes |
|---|---|---:|---|
| Qwen2.5-3B-Instruct | Q4_K_M | **~26.6** | n=5 paired median, locked default config (predec + vocab-prune-32K + Q4K-LM-head + ffn_down-Q4K) |
| DeepSeek-V2-Lite-Chat | Q4_K_M | **~17** | TRIALS=4 TOKENS=24 coexist, 95% CI [16.6, 18.0] |
| Mixtral-8x7B-Instruct-v0.1 | Q3_K_M | **~0.1** | functional, SSD-bandwidth-limited on 18 GB |

llama.cpp Metal on Qwen-3B-Q4_K_M on the same hardware lands around **50 dec_tps**; dismantle's gap is **1.88×** as of 2026-05-26 (first sub-2× measurement on M3 Pro). For DeepSeek-V2-Lite the gap is roughly 3×. dismantle prioritizes a small, auditable Rust codebase over matching every C++ kernel optimization. The gap is honest engineering work; it's not a fundamental architectural limit. See [reports/v1.1.0_architecture_audit.md](reports/v1.1.0_architecture_audit.md) for the bandwidth/utilization breakdown and the composition decision matrix in `memory/composition_decision_matrix_2026_05_26.md` for the Qwen-3B optimization path.

## Requirements

- Apple Silicon Mac (M1, M2, M3, or M4)
- Rust stable
- ~12 GB free memory for DeepSeek-V2-Lite Q4_K_M (model + KV cache)
- ~16 GB free disk + ~14 GB RAM for Mixtral 8×7B Q3_K_M

## Build

```sh
git clone https://github.com/joshuahickscorp/dismantle.git
cd dismantle
cargo build --release --workspace
# Binary: target/release/dismantle
```

## Get a model

```sh
./tools/fetch-model.sh        # downloads DeepSeek-V2-Lite Q4_K_M (~9.7 GB)
./tools/fetch-mixtral.sh      # downloads Mixtral 8×7B Q3_K_M (~16 GB)
```

Or pass any GGUF file via `--weights`. The architecture is detected from metadata.

## Usage

**Check fit before loading:**

```sh
dismantle doctor --weights models/deepseek-v2-lite-q4.gguf
```

**Pick the fastest kernels for your machine** (run once, takes 1–2 min):

```sh
dismantle autotune \
  --weights models/deepseek-v2-lite-q4.gguf \
  --out profiles/my-mac.json
```

**Generate:**

```sh
dismantle generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/my-mac.json \
  --prompt "Once upon a time" \
  --max-new-tokens 256
```

**Serve as OpenAI-compatible HTTP API:**

```sh
dismantle serve \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/my-mac.json \
  --addr 127.0.0.1:8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "Write a haiku about Metal kernels."}],
    "max_tokens": 64
  }'
```

**Reproduce the perf numbers in this README:**

```sh
TRIALS=4 TOKENS=24 bash tools/bench/coexist_bench.sh
```

The script reports median, 95% confidence interval, and IQR. Run with `TRIALS=6 TOKENS=64` for a tighter authoritative number (~15 min total). See [tools/bench/README.md](tools/bench/README.md) for the standardized bench parameter conventions.

## Mixtral 8×7B support

Mixtral Q3_K_M is supported as a secondary target. See [docs/mixtral.md](docs/mixtral.md) for fetch + run instructions and expected throughput. Performance is limited by SSD bandwidth on 18 GB machines (expert weights page-fault from disk between layers); 32+ GB machines run faster because more weights stay resident.

## Project status

**Pre-v2.0, active development.** v2.0 launch focuses on shipping the engine at its current honest performance with a clean, auditable codebase. Future work toward llama.cpp-class throughput (Apple Neural Engine integration, true K-parallel batched verify for speculative decoding) is post-v2.0 and not gating this release.

## License

MIT. See [LICENSE](LICENSE).
