#!/usr/bin/env python3.12
"""MLX perplexity — the third engine for the rigorous 3-way bench (Hawking vs llama vs MLX).
Computes relative degradation (4-bit ppl / bf16 ppl) so MLX is comparable to the others
(each engine vs its own f16, cancels harness/tokenizer differences).

Usage: mlx_ppl.py <mlx-model-dir> [text-file]
Prep:  python3.12 -m mlx_lm convert --hf-path <hf> -q --q-bits 4 --mlx-path <out-4bit>
       python3.12 -m mlx_lm convert --hf-path <hf>      --mlx-path <out-bf16>
"""
import sys, math
import mlx.core as mx
from mlx_lm import load

path = sys.argv[1]
text = open(sys.argv[2], errors="ignore").read() if len(sys.argv) > 2 else \
    "The science of operations, as derived from mathematics more especially, is a science of itself."

model, tok = load(path)
ids = mx.array(tok.encode(text)[:2048])
logits = model(ids[None])[0]                      # [T, V]
logp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
tgt = ids[1:]
nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).mean()
print(f"{math.exp(nll.item()):.4f}")
