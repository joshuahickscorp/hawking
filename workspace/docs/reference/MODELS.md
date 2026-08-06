# Verified models & quant formats

What has actually been run and checked on this hardware (Apple M3 Pro, 18 GB), versus what merely has a code path. Be honest with yourself about the tiers before depending on one:

- ✅ **Verified** — a parity or quality gate covers it and it produces correct output here.
- 🟡 **Runs** — it loads and generates, but output quality isn't gated in this environment.
- ⬜ **Untested** — the loader recognizes the architecture, but it hasn't been exercised end-to-end here.

## Dense

| Model | Quant | Status | Evidence |
|---|---|---|---|
| Qwen2.5-3B-Instruct | Q4_K_M | ✅ Verified | primary target; greedy + multiseq token parity gates and the W4A8 cosine-vs-f32 quality gate; generates correct, coherent output |
| Qwen2.5-0.5B-Instruct | Q4_K_M | ✅ Verified | CPU↔Metal greedy parity, 12/12 leading token ids identical |
| Qwen2.5-1.5B / 7B | Q4_K_M | 🟡 Runs | same dense path; tuned profiles present; not gated here |
| Llama 3.x | Q4_K_M | ⬜ Untested | architecture detected; no gate run in this environment |
| Mistral | Q4_K_M | ⬜ Untested | architecture detected; no gate run |
| Gemma 2 | Q4_K_M | ⬜ Untested | architecture detected; no gate run |
| Phi-3 / 3.5 | Q4_K_M | ⬜ Untested | `model/phi3.rs` present; no gate run |

## MoE

| Model | Quant | Status | Evidence |
|---|---|---|---|
| DeepSeek-V2-Lite-Chat | Q4_K_M | 🟡 Runs | MLA attention path; historically measured ≈17 tok/s; integration + CPU-parity gates exist but **skip** without the GGUF on disk |
| Mixtral-8×7B-Instruct | Q3_K_M | 🟡 Runs (impractical) | loads, but SSD-bandwidth-limited (≈0.1 tok/s) on an 18 GB machine |
| Qwen3-MoE | — | ⬜ Untested | architecture detected; no gate run |

## Quant formats

| Format | Status |
|---|---|
| Q4_K_M | ✅ Verified — the tuned decode path, parity-gated against the CPU reference |
| Q6_K | 🟡 Runs — used for some tensors (LM head / ffn_down variants), kernel-parity tested |
| Q3_K_M | 🟡 Runs — Mixtral path; bandwidth-bound |
| Q8_0 / f16 | 🟡 Runs — reference / fallback paths |
| Others (Q2_K, Q5_K, IQ\*) | ⬜ Untested |

## Verify a model yourself

```sh
hawking doctor   --weights <model.gguf>                 # fit + metadata
hawking generate --weights <model.gguf> --prompt "..." --max-new-tokens 64
```

To add a parity gate for a new model, follow `crates/hawking-core/tests/greedy_token_only_parity.rs` — it skips cleanly when the weights are absent, so it stays CI-safe.
