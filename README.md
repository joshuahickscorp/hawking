# Hawking

Hawking is a from-scratch Rust and Metal LLM inference engine for Apple
silicon. It loads GGUF models through mmap, runs hand-written kernels, and
exposes a CLI plus an OpenAI-compatible local server. The runtime has no Python,
`llama.cpp`, BLAS, or MPSGraph dependency.

The project is evidence-first: correctness precedes speed, approximate paths
carry explicit quality gates, and low-bit claims use whole-artifact physical
bpw rather than nominal payload bits.

## Start here

- [Handbook](docs/README.md): architecture, models, HTTP API, and document map
- [Operations](docs/OPERATIONS.md): builds, validation, benchmarks, receipts,
  Doctor status, and recovery
- [Research](docs/RESEARCH.md): evidence rules, current lanes, and killed levers
- [Roadmap](docs/plans/ROADMAP.md): active sequencing and HIDE product contract
- [STRAND](docs/STRAND.md): low-bit execution and promotion

Dated reports and retired instructions remain available in Git history. The
mapping from removed Markdown paths to their canonical replacements is
[`docs/markdown_redirects.json`](docs/markdown_redirects.json).

## Build

Requirements are Apple silicon, Rust 1.80 or newer, and Xcode Command Line
Tools.

```sh
git clone https://github.com/joshuahickscorp/hawking.git
cd hawking
python3 tools/hawking_packs.py fetch
python3 tools/hawking_packs.py hydrate
python3 tools/hawking_packs.py verify
cargo build --release --workspace
```

The CLI is written to `target/release/hawking`.
For a network-free build, preseed the sibling `../hawking-pack-cache` directory
or set `HAWKING_PACK_CACHE`, then replace `fetch` with `fetch --offline`.

## Model and generation

Fetch a pinned profile or provide your own GGUF:

```sh
python3 tools/ops.py model fetch deepseek-v2-lite-q4

./target/release/hawking doctor \
  --weights models/deepseek-v2-lite-q4.gguf

./target/release/hawking generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --prompt "Write a Rust function that reverses a linked list." \
  --max-new-tokens 256
```

Tune a device/model pair and pass the resulting profile explicitly:

```sh
./target/release/hawking autotune \
  --weights models/deepseek-v2-lite-q4.gguf \
  --out profiles/my-mac.json
```

## Local HTTP server

```sh
./target/release/hawking serve \
  --weights models/deepseek-v2-lite-q4.gguf \
  --addr 127.0.0.1:8080

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

Bind to loopback unless an authenticated reverse proxy supplies the external
security boundary.

## Validation and benchmarks

```sh
python3 tools/hawking_packs.py validation
cargo fmt --all -- --check
CARGO_TARGET_DIR=/tmp/hawking-target cargo test --workspace --no-run
python3 tools/ops.py selftest

python3 tools/ops.py bench profiles
TRIALS=4 TOKENS=24 python3 tools/ops.py bench run coexist
python3 tools/ops.py bench run paired -- \
  --label example --env-a "FEATURE=0" --env-b "FEATURE=1"
```

Performance claims require matched baselines on the same machine and resource
envelope. See the operations guide before publishing a result.

## License

MIT. See [LICENSE](LICENSE).
