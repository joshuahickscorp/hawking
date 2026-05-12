# dismantle

Pure-Rust + Apple Metal inference engine for MoE language models on Apple Silicon. Single binary, no Python in the runtime, no llama.cpp dependency. Loads GGUF weights via mmap and runs them through hand-rolled Metal compute kernels.

Currently optimized for **DeepSeek-V2-Lite Q4_K_M** (16B params, 2.4B active per token). v1.0.0 pivots toward MoE memory differentiation while keeping the current ~19.5 dec_tps V2-Lite path stable.

## What's distinctive

- **MoE-aware expert offloading** — first Apple Silicon inference engine to fit Mixtral 8x7B in 18GB by paging cold expert weights to OS cache while keeping active experts hot
- **Pure Rust + Metal** — single binary, no Python in runtime
- **Reproducible kernel autotune** — deterministic per-machine profile selection

## Requirements

- Apple Silicon Mac (M1, M2, M3, or M4)
- Rust stable
- ~12 GB free memory for DeepSeek-V2-Lite Q4_K_M (model + KV cache)

## Build

```sh
git clone https://github.com/joshuahickscorp/dismantle.git
cd dismantle
cargo build --release --workspace
# Binary: target/release/dismantle
```

## Get a model

```sh
./tools/fetch-model.sh   # downloads DeepSeek-V2-Lite Q4_K_M (~9.7 GB)
```

Or pass any GGUF file via `--weights`. Architecture is detected from metadata.

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

## Status

v1.0.0 launch candidate, active development. DeepSeek-V2-Lite generation is stable; Mixtral 8x7B support is preview-scaffolded and tracked in [docs/mixtral.md](docs/mixtral.md).

## License

MIT. See [LICENSE](LICENSE).
