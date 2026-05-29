#!/usr/bin/env python3
"""build_frozen_gguf — build an Eagle5 frozen-weights npz from the GGUF the
runtime actually serves.

THE FIX (2026-05-29): the Eagle5 head's baseline is
`argmax(RMSNorm(residual, output_norm) @ lm_head)`. For the head's drafts to
match the runtime verifier, `lm_head`/`token_embd`/`output_norm` MUST be the
exact weights the runtime serves — i.e. the GGUF's Q6_K-dequantized tensors,
NOT an HF fp16 export. The old `eagle4/qwen3b_frozen.npz` was built from HF
fp16 and differed from the GGUF by up to 0.27 per element, which completely
flips the argmax and tanks real spec-decode acceptance to ~0%. Rebuilding
frozen weights from the GGUF dequant lifts the offline lens ceiling from 0%
to ~85-100%.

Emits the trainer's frozen-npz schema:
  * token_embd : fp16 (hidden, vocab)
  * lm_head    : fp16 (hidden, vocab)   (tied to token_embd when the GGUF has
                 no separate output.weight, as in Qwen2.5)
  * output_norm: fp32 (hidden,)

Usage
-----
    python3 tools/orchestrator/build_frozen_gguf.py \
        --gguf models/qwen2.5-3b-instruct-q4_k_m.gguf \
        --out  eagle4/q3b_frozen_gguf.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    from gguf import GGUFReader, dequantize
except ImportError:
    sys.stderr.write("needs gguf: pip install gguf\n")
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(prog="build_frozen_gguf")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    tensors = {t.name: t for t in r.tensors}

    def deq(name: str) -> np.ndarray:
        t = tensors[name]
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    if "token_embd.weight" not in tensors:
        sys.stderr.write("token_embd.weight not found in GGUF\n")
        return 1
    # token_embd.weight is (vocab, hidden) in GGUF; frozen npz wants (hidden, vocab).
    te = deq("token_embd.weight")            # (vocab, hidden)
    tok_embd = te.T.copy().astype(np.float16)  # (hidden, vocab)

    # lm_head: prefer a separate output.weight; else tie to token_embd.
    if "output.weight" in tensors:
        lm = deq("output.weight")            # (vocab, hidden)
        lm_head = lm.T.copy().astype(np.float16)
        tied = False
    else:
        lm_head = tok_embd.copy()
        tied = True

    if "output_norm.weight" not in tensors:
        sys.stderr.write("output_norm.weight not found in GGUF\n")
        return 1
    output_norm = np.array(tensors["output_norm.weight"].data).astype(np.float32)

    hidden, vocab = tok_embd.shape
    np.savez(args.out, token_embd=tok_embd, lm_head=lm_head, output_norm=output_norm)
    print(f"[build_frozen_gguf] wrote {args.out}")
    print(f"  hidden={hidden} vocab={vocab} lm_head_tied={tied}")
    print(f"  output_norm: mean={output_norm.mean():.4f} std={output_norm.std():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
