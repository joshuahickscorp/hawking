# dismantle

Frontier Apple Silicon MoE inference in Rust + Metal.

dismantle is not a wrapper around llama.cpp or MLX. It is a GGUF-native
Mac inference engine built around the part of Mixture-of-Experts models
that generic runtimes leave exposed: expert routing, quantized expert
GEMV, unified-memory residency, and launch overhead.

The near-term target is practical and specific: make DeepSeek-V2-Lite Q4
usable on an M3 Pro with 18 GB unified memory, while pushing toward the
first FlashMoE-style Metal path for Apple Silicon.

## Status

dismantle 0.1.0 ships with three paths on Apple Silicon, all
parity-attested at atol < 1e-3.

### Measured perf — M3 Pro 18 GB

| Model | Path | dismantle dec_tps | llama.cpp Metal dec_tps | ratio |
|---|---|---:|---:|---:|
| DeepSeek-V2-Lite Q4_K_M | indexed-no-pack-one-cb (default) | 1.61 | pending | — |
| Qwen2.5-3B Q4_K_M | dense | 1.28 | pending | — |

(All numbers: 3 trials × 64 tokens, greedy temp=0. dismantle measured
2026-05-02 post-layered-MoE wedges; ratio is dismantle/llama.cpp.
llama.cpp comparison pending: `ggml 0.10.0` Metal tensor API disabled
on M3 Pro — benchmark deferred to v0.2 with a verified Metal build.)

The DeepSeek MoE path runs FASTER than the dense Qwen path on the
same hardware — the layered batched + no-pack + one-command-buffer
MoE wedges deliver ~4.7× over the pre-wedge baseline (0.34 tok/s).

The strict single-kernel fused FlashMoE is shipped opt-in
(`moe_schedule: "single-kernel"`) for prefill/batch use; on
single-token decode it's ~90× slower than the batched default due to
redundant intermediate compute. v0.2 ships the two-stage redesign.

`generate`, `serve`, `bench`, `batch-hash`, `doctor`, `autotune`, and
`shader-hash` subcommands are wired. Metal parity gates cover RMSNorm,
Q4/Q6/Q8 quant GEMV, MLA paths, weight pinning, batched MoE, no-pack
indexed MoE, and the fused single-kernel MoE family.

## M3 Pro 18 GB Profile

Hero hardware:

- Apple M3 Pro, 14-core GPU class
- 18 GB unified memory
- DeepSeek-V2-Lite Q4 GGUF: `9.7 GiB` on disk in this workspace
- Hero shape: 15.7B total parameters, about 2.4B active per token,
  MLA attention, 2 shared experts, top-6 of 64 routed experts

The memory strategy is to keep GGUF weights mmap-backed, expose the whole
mmap to Metal as a no-copy buffer, and pass tensor byte offsets into
kernels. That avoids per-token expert packing and avoids duplicating the
full 9.7 GB model just to make Metal see it.

Check your local fit:

```sh
target/release/dismantle doctor \
  --weights models/deepseek-v2-lite-q4.gguf \
  --max-seq-len 4096
```

## Why This Is Interesting

FlashMoE showed the right direction: keep MoE scheduling and expert work
GPU-resident, reduce launch overhead, and avoid CPU-managed routing gaps.
dismantle ports that idea to Apple Silicon in stages:

| Stage | State | What changes |
|---|---:|---|
| CPU reference | green | correctness-first DeepSeek path |
| Metal quant GEMV | green | Q4_K/Q6_K/Q8_0 dequant fused into GEMV |
| Batched MoE | green | selected experts run as batched GEMVs |
| No-pack MoE | green | route IDs index fused GGUF expert tensors directly |
| One-command-buffer MoE | green | routed/shared MoE kernels commit and wait once |
| Strict fused FlashMoE | shipped (opt-in only) | single-kernel `moe_block_fused_v2lite{,_indexed}` — correct at atol<1e-3, decode-redundant; v0.2 redesign two-stage |
| Two-stage fused MoE | next (v0.2) | persist intermediate to device memory; eliminate redundant compute |

The point is to be usable while still pushing the edge. The stable path
must keep `generate` and `serve` working; the experimental path can chase
the single-kernel public claim once the layered path is measured.

## Deterministic Moonshot Controls

The research path is profile-driven so overnight work is reproducible
instead of anecdotal. `autotune` emits a deterministic profile tied to
the model layout, Metal shader hash, and local GPU name:

```sh
target/release/dismantle autotune \
  --weights models/deepseek-v2-lite-q4.gguf \
  --profile m3-pro-18gb \
  --max-hours 8 \
  --out profiles/deepseek-v2-lite-q4.m3pro18.json
```

It also writes `profiles/deepseek-v2-lite-q4.m3pro18.jsonl`, a
line-delimited candidate log that can be diffed across long runs.

Use a profile explicitly when benchmarking or serving:

```sh
target/release/dismantle bench \
  --backend dismantle \
  --suite decode \
  --weights models/deepseek-v2-lite-q4.gguf \
  --kernel-profile profiles/deepseek-v2-lite-q4.m3pro18.json \
  --trace-json /tmp/dismantle_trace64.json \
  --trials 1 \
  --max-new-tokens 64
```

Exact speculative decode is opt-in and currently correctness-first:
`--speculate exact-shared --verify-window 4` requires greedy
`temperature=0` and preserves the verifier token stream.

The 50-prompt token regression can run through the same profiled path:

```sh
DISMANTLE_KERNEL_PROFILE=profiles/deepseek-v2-lite-q4.m3pro18.json \
DISMANTLE_SPECULATE=exact-shared \
DISMANTLE_VERIFY_WINDOW=4 \
tools/haul/token-regression.sh tests/golden/_phase2_token_baseline_50.hashes
```

## Quick Start

Build:

```sh
cargo build --release --workspace
```

Generate:

```sh
target/release/dismantle generate \
  --weights models/deepseek-v2-lite-q4.gguf \
  --prompt "To be or not to be" \
  --max-new-tokens 3 \
  --temperature 0 \
  --max-stall-ms 240000
```

Run a fast decode bench:

```sh
target/release/dismantle bench \
  --backend dismantle \
  --suite decode \
  --weights models/deepseek-v2-lite-q4.gguf \
  --trials 1 \
  --max-new-tokens 16 \
  --json /tmp/dismantle_decode16.json
```

Serve an OpenAI-compatible endpoint:

```sh
target/release/dismantle serve \
  --weights models/deepseek-v2-lite-q4.gguf \
  --addr 127.0.0.1:8080
```

## Correctness Gates

The core floor before publishing any speed claim:

```sh
cargo test --workspace --lib
cargo test --release --test phase1_kernel_parity
cargo test --release --test phase2_mla_metal_parity
cargo test --release --test phase2_weight_pinning_parity
cargo test --release --test phase2_moe_block_batched_parity
```

The parity regime is intentionally tight: max absolute diff below `1e-3`
for Metal-vs-reference kernel paths. Token-output baselines are kept
separate because tiny fp16 shifts can flip near-tied argmaxes.

## Competitive Posture

MLX is the usability and Apple Silicon runtime bar. llama.cpp is the CLI,
server, and benchmark-reproducibility bar. dismantle competes by being
MoE-specific:

- GGUF-native Rust engine, not a subprocess wrapper.
- Quantized expert weights stay quantized into the Metal GEMV.
- Routed/shared experts are batched and now indexed in-place from fused
  tensors.
- The public FlashMoE claim is reserved for the strict fused kernel after
  it is correct and measured.

See [docs/competitive_audit.md](docs/competitive_audit.md) and
[ROADMAP.md](ROADMAP.md) for the broader landscape and longer wedge list.

## What This Is Not

- Not CUDA, ROCm, Vulkan, or MPSGraph. Apple Silicon is the v0.1 target.
- Not a training stack.
- Not a polished chat UI.
- Not a universal dense-model replacement yet, though the Qwen dense path
  exists as the practical daily-driver direction beside the MoE hero path.

## Credits

- DeepSeek-AI for the V2/V3 architecture and weights.
- Georgi Gerganov and the llama.cpp community for the GGUF ecosystem and
  the usability standard local engines are measured against.
- Apple's MLX team for showing how good unified-memory ML on Apple
  Silicon can feel.

## License

MIT. See [LICENSE](LICENSE).
