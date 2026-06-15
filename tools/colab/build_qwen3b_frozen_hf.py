#!/usr/bin/env python3
"""Build the Qwen2.5-3B frozen baseline .npz from Hugging Face weights.

The Eagle5 trainer expects:
  token_embd  float16 (hidden, vocab)
  lm_head     float16 (hidden, vocab)
  output_norm float32 (hidden,)

This Colab path avoids requiring a local GGUF. It downloads the same HF model
used by calibration and writes atomically so interrupted cells do not leave a
half-written frozen baseline.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def _write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.out.exists() and not args.force:
        print(
            f"[frozen-hf] exists: {args.out} "
            f"({args.out.stat().st_size / 1e9:.2f} GB); skipping",
            flush=True,
        )
        return 0

    print(f"[frozen-hf] loading {args.model} on CPU", flush=True)
    model_kwargs = {
        "dtype": torch.float16,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    except TypeError:
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()

    embed_w = model.model.embed_tokens.weight.detach()
    lm_w = model.lm_head.weight.detach()
    tied = embed_w.data_ptr() == lm_w.data_ptr()

    embed = embed_w.to(torch.float16).cpu().numpy().T.copy()
    if tied:
        lm_head = embed.copy()
    else:
        lm_head = lm_w.to(torch.float16).cpu().numpy().T.copy()
    output_norm = model.model.norm.weight.detach().to(torch.float32).cpu().numpy().copy()

    if embed.ndim != 2 or lm_head.shape != embed.shape:
        raise RuntimeError(
            f"bad frozen table shapes: embed={embed.shape} lm_head={lm_head.shape}"
        )
    if output_norm.shape != (embed.shape[0],):
        raise RuntimeError(
            f"bad norm shape: norm={output_norm.shape} hidden={embed.shape[0]}"
        )

    _write_npz_atomic(
        args.out,
        {
            "token_embd": embed,
            "lm_head": lm_head,
            "output_norm": output_norm,
        },
    )
    print(
        f"[frozen-hf] wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB) "
        f"embed={embed.shape} tied={tied}",
        flush=True,
    )

    del model, embed, lm_head, output_norm
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
