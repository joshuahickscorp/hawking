# Resident Genesis body — measured 2026-08-16

One process holds the Qwen3.8 weight set and serves proposals. The loop no
longer pays a load on every `genesis_proposes()`. This is a load-time /
residency win. Concurrent decode ceiling remains 1; do not add sessions
to chase tokens/s.

## This run (DIRTY_ENGINEERING, cache-hot)

A preceding lane had just uploaded `uniform-q4-v1`, so these walls are
**not** the established ~50 s box-cold load.

| step | wall | notes |
|---|---:|---|
| greedy oneshot (today's loop) | 4.649 s | load + 4-token generate + exit |
| resident `load_ns` | 4.908 s | `Qwen38HybridWeights::load` + `attach` |
| serve 1 / 2 / 3 | 0.939 / 0.917 / 0.922 s | client RPC; generate 0.635 / 0.521 / 0.524 s |

- `load_count = 1` across three serves, same pid 55576
- greedy ids identical: `[248068, 198, 760, 1156]`, 0 fallbacks
- `resident_weight_bytes = 14,297,675,776` (matches the shared-sessions receipt)
- RSS after load 15.512 GB; phys_footprint 15.386 GB
- stopfile: process exited 0, `kill 0` dead, health None
- fallback: service down, `genesis_proposes()` still returned text

## How to run it

```
CARGO_TARGET_DIR=/tmp/genesis-resident-target \
  cargo build --release --manifest-path tools/agentos/genesis_body/Cargo.toml

python3 tools/agentos/genesis_resident.py serve \
  --artifact-root workspace/campaign/records/runs/qwen38-27b/uniform-q4-v1 \
  --tokenizer workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json \
  --max-seq-len 4096
```

`ascent_daemon.genesis_proposes()` talks to the socket first. If the body
is not live it shells out to `ascension_qwen38_hybrid_greedy` as before.
`touch workspace/ops/GENESIS_STOP` still stops the launchd loop; the body
watches the same path and exits 0.

Do not start a second body. Separate processes do not share artifact pages.
