# dismantle

Apple Silicon MoE inference engine in Rust + Metal. Runs DeepSeek-V2 and
Qwen2.5 GGUF models with a custom Metal kernel stack — two-stage fused MoE,
Metal MLA decode, layer command-buffer batching, and a decode-arena buffer
pool. MIT.

## Install

### Pre-built binary (Apple Silicon Mac)

Download `dismantle-v0.2.0-aarch64-apple-darwin.tar.gz` from the
[v0.2.0 release](https://github.com/joshuahickscorp/dismantle/releases/tag/v0.2.0),
extract, and put `dismantle` somewhere on your `$PATH`.

### From source

```sh
git clone https://github.com/joshuahickscorp/dismantle.git
cd dismantle
cargo build --release --workspace
# Binary lands at target/release/dismantle
```

Requires Rust stable + Apple Silicon Mac (M1/M2/M3/M4).

## Quick Start

Get the model (~9.7 GB):

```sh
./tools/fetch-model.sh
```

Generate text:

```sh
dismantle generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --prompt "Once upon a time" \
  --max-new-tokens 32
```

Serve an OpenAI-compatible HTTP endpoint:

```sh
dismantle serve \
  --weights models/deepseek-v2-lite-q4.gguf \
  --addr 127.0.0.1:8080 &

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V2-Lite-Chat",
    "messages": [{"role": "user", "content": "Write a haiku about Metal kernels."}],
    "max_tokens": 64
  }'
```

See [docs/serve.md](docs/serve.md) for full API reference and streaming examples.

Check whether the model fits in memory before loading it:

```sh
dismantle doctor --weights models/deepseek-v2-lite-q4.gguf
```

## Supported models

Tested:
- **DeepSeek-V2-Lite-Chat Q4\_K\_M** — MoE hero model (16B total / 2.4B active per token)
- **Qwen2.5-3B-Instruct Q4\_K\_M** — dense path

Should work: other Qwen2 / Qwen2.5 GGUFs. Pass any GGUF via `--weights`; the
architecture is auto-detected from metadata.

## Performance

M3 Pro 18 GB, DeepSeek-V2-Lite Q4\_K\_M, greedy temp=0:

| Version | Backend | dec\_tps | Notes |
|---|---|---:|---|
| v0.2.0 | dismantle | **TBD¹** | two-stage MoE + Metal MLA + layer-CB + decode-arena |
| v0.1.0 | dismantle | **1.61** | 3 trials × 64 tokens, layered batched MoE |
| — | llama.cpp b9000 | **59.6** | tg16, ngl 99, ggml 0.10.2 Metal |

¹ Bench deferred — slm training was running concurrently during v0.2.0 finalization.
Per-wedge smoke tests all passed (coherent output). Dedicated bench window
queued; see [docs/v0.2.0\_closeout.md](docs/v0.2.0_closeout.md).

v0.2.0 ships four performance wedges over v0.1.0: two-stage fused MoE
(eliminates single-kernel decode-redundant intermediate compute), Metal MLA
decode (replaces CPU path), layer-CB (batches mla_decode + o_proj into one
command buffer), and decode-arena (pre-allocated Metal buffer pool).

Full story in [docs/v0.2.0\_closeout.md](docs/v0.2.0_closeout.md).

## What's next

v0.3 targets: measured bench against llama.cpp Metal + MLX with the full
harness, persistent FlashMoE research variant, prefill path. See [ROADMAP.md](ROADMAP.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build, test, and parity-gate instructions.

## Credits

- DeepSeek-AI for the V2/V3 architecture and weights.
- Georgi Gerganov and the llama.cpp community for the GGUF ecosystem and the
  usability standard local engines are measured against.
- Apple's MLX team for demonstrating unified-memory ML on Apple Silicon.

## License

MIT. See [LICENSE](LICENSE).
