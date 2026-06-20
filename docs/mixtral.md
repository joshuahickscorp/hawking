# Mixtral 8×7B support

hawking supports running Mixtral 8×7B GGUF models on Apple Silicon. The
**Q3_K_M** quantization is the recommended target for 18 GB Macs because the
on-disk model (~16 GB) fits with enough headroom for the KV cache and
activations.

## Status

**Functional but slow on memory-constrained hardware.** The forward path is
correct and produces coherent English output. Decode throughput is currently
SSD-bandwidth-limited on 18 GB machines because expert weights page-fault
from disk between layers (only 2 of 8 experts are active per token, but
routing varies between tokens faster than the OS can speculatively cache).
On 32+ GB machines the entire model fits in RAM and throughput is
substantially higher.

| machine | dec_tps (Mixtral 8×7B Q3_K_M) | notes |
|---|---:|---|
| M3 Pro 18 GB | ~0.1 | SSD-bandwidth-limited |
| M-series, 32+ GB | not yet measured | expected to be RAM-resident |

For benchmarking purposes on 18 GB hardware, the Q4_K_M quantization
(~26 GB) does not fit and is not supported on this configuration.

## Model

Recommended source:
[TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF).

Use the included fetch script:

```sh
./tools/fetch-mixtral.sh
```

Or download manually to `models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf`.

## Run

```sh
./target/release/hawking generate \
    --weights models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf \
    --max-routed-expert-ram-mb 14000 \
    --prompt "Once upon a time" \
    --max-new-tokens 32
```

The `--max-routed-expert-ram-mb` flag caps expert weight residency. On
18 GB Macs, 14000 leaves ~4 GB for everything else (activations,
KV cache, OS overhead). On larger Macs you can raise or omit this flag.

## What hawking does for Mixtral specifically

- **Architecture detection** — auto-recognizes Mixtral's standard
  MHA + GQA + RoPE structure (32 layers, hidden=4096, intermediate=14336,
  32 Q heads, 8 KV heads, top-2 routing, no shared expert).
- **Tokenizer** — llama.cpp-compatible SentencePiece BPE with
  byte-fallback (verified to produce identical token sequences to
  llama-tokenize).
- **Q3_K kernel** — Metal GEMV implementation matches scalar reference
  at atol=1e-2 (tested in `tests/v1_1_q3_k_parity.rs`).
- **Mixed-quant routing** — Q4_K_M attention projections, Q5_K LM head,
  Q8_0 K/V dequant — each path uses its dedicated kernel.
- **Expert cache** — `ExpertCache` tracks per-expert per-layer access
  frequency over a rolling window; `--max-routed-expert-ram-mb` triggers
  `posix_madvise(MADV_DONTNEED)` on cold expert pages to keep RAM under
  the budget. Page-faults reload pages on next access.

## Known limitations

- **Throughput is SSD-bound on 18 GB.** Expect ~0.1 dec_tps. Acceptable for
  "does it work" experiments and structured short responses, not for
  interactive conversation on 18 GB hardware.
- **Long prompts are slow at prefill** — same root cause (cold expert pages
  during prefill compute).
- **Streaming serving is functional but not recommended** on 18 GB. Use
  the OpenAI-compatible HTTP API on 32+ GB machines for usable serving.

## Future work

Improving Mixtral 18 GB throughput requires addressing the expert-weight
SSD bandwidth ceiling. Options under consideration for post-v2.0:

- Predictive expert prefetch — speculate the next layer's active experts
  during current layer compute
- Lower-precision quantization (Q2_K) to fit more experts in RAM
- Persistent kernel for the MoE block to avoid mmap thrash

These are not gating v2.0 release. The current support ships as a
working baseline.
