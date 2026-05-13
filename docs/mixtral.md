# Mixtral 8x7B

Status: **functional but unoptimized** as of v1.0.4. The model loads on 18 GB
Macs with `--max-routed-expert-ram-mb 8000`, the GPU path activates, the
SentencePiece tokenizer produces correct token IDs matching llama.cpp, and
generation produces coherent English. Current decode speed is ~0.12 dec_tps;
v1.0.5+ targets the perf push toward ≥3 dec_tps.

Example output for `"Once upon a time"`:
```
Once upon a time, there was a young girl who was ...
```

## Architecture support
- 32 layers, hidden=4096, intermediate=14336
- Standard MHA + GQA (32 Q heads, 8 KV heads, head_dim=128)
- RoPE θ = 1,000,000
- 8 routed experts, top-2, no shared expert
- Uniform Q4_K_M weights (attn projections + experts); Q5_K LM head; Q8_0 K/V
- SentencePiece tokenizer (llama.cpp greedy score-merge port, byte-fallback)

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
