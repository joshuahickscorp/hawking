# Mixtral 8x7B Preview

Status: preview, not yet functional in v1.0.0.

Phase 2 lands Mixtral architecture detection and tokenizer scaffolding, plus
the Phase 1 expert-cache surface that the full engine will use for cold expert
eviction. The standard-MHA forward path, 8-expert top-2 MoE execution, and real
18 GB smoke demo are intentionally deferred to v1.0.1.

## Model Source

Use a Q4_K_M GGUF export such as
[TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF).

Expected local filename:

```bash
models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
```

## Planned 18 GB Command

```bash
target/release/dismantle generate \
  --weights models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
  --prompt "Once upon a time" \
  --max-new-tokens 64 \
  --max-routed-expert-ram-mb 8000
```

In v1.0.0 this returns a clear preview error. v1.0.1 should wire the
`MixtralEngine` skeleton to standard MHA attention, LLaMA RoPE, uniform Q4_K_M
matmuls, and Phase 1's `ExpertCache` ranges so roughly two of eight experts per
layer stay hot while colder expert pages are returned to the OS cache.
