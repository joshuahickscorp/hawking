#!/usr/bin/env python3
"""Minimal Eagle5 v2 corpus builder for Colab GPU.

Loads DeepSeek-V2-Lite-Chat (HF native, fp16), runs ultrachat sequences
through with SINGLE-layer activation capture (default layer 25 — what
eagle5_train.py actually reads), writes int8-quantized parquet shards.

Stripped of all the complexity of the legacy build_corpus.py:
  - No MPS/CPU/multi-device branching (CUDA only, this is Colab)
  - No bitsandbytes 4-bit (modern Colab GPUs fit V2-Lite at fp16)
  - No offloading
  - No per-layer×27 captures (eagle5_train uses ONE layer)
  - No expert/gate hooks (eagle5_train uses residual + intermediate only)
  - Aggressive per-batch torch.cuda.empty_cache()
  - Atomic shard writes (idempotent --skip-existing resume)

Run via:
  python colab/corpus_simple.py \
    --out /content/v2_lite_corpus \
    --max-sequences 100000 \
    --batch-size 8 \
    --max-tokens 2048

~150 lines. If something breaks, easy to debug.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def quantize_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 quantize. Returns (int8 bytes, scale)."""
    arr = arr.astype(np.float32)
    max_abs = float(np.abs(arr).max()) if arr.size else 0.0
    if max_abs < 1e-8:
        return np.zeros(arr.shape, dtype=np.int8), 0.0
    scale = max_abs / 127.0
    q = np.round(arr / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--split", default="train_sft")
    p.add_argument("--max-sequences", type=int, default=100000)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--capture-layer", type=int, default=25)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Lazy-import pyarrow only after argparse so --help is instant
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"[corpus] loading {args.model} (native HF, fp16, sdpa attention)",
          flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=False,
        device_map="cuda",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval()
    print(f"[corpus] loaded in {time.time() - t0:.1f}s", flush=True)

    # Single-layer hooks (way less memory than legacy all-layers capture)
    L = args.capture_layer
    layer = model.model.layers[L]
    captured: dict[str, torch.Tensor] = {}

    def _residual_hook(_m, _i, out):
        captured["residual"] = (out[0] if isinstance(out, tuple) else out).detach()

    def _intermediate_hook(_m, _i, out):
        captured["intermediate"] = (out[0] if isinstance(out, tuple) else out).detach()

    layer.register_forward_hook(_residual_hook)
    mlp = layer.mlp
    target = mlp.experts if hasattr(mlp, "experts") else mlp
    target.register_forward_hook(_intermediate_hook)

    # Resume from existing shards (atomic; sized-based dedup)
    existing = sorted(args.out.glob("shard_*.parquet"))
    shard_idx = len(existing)
    yielded = shard_idx * args.shard_size
    print(f"[corpus] resume: {yielded} seqs done, starting shard {shard_idx}",
          flush=True)
    if yielded >= args.max_sequences:
        print("[corpus] already at target — nothing to do", flush=True)
        return 0

    print(f"[corpus] streaming {args.dataset}[{args.split}]", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    ds_iter = iter(ds)
    # Skip past already-processed sequences (cheap; just iter advance)
    for _ in range(yielded):
        next(ds_iter, None)

    pbar = tqdm(total=args.max_sequences, initial=yielded, desc="seqs",
                file=sys.stderr, mininterval=2.0)
    buf: list[dict] = []

    while yielded < args.max_sequences:
        # Gather a batch of texts
        texts: list[str] = []
        for _ in range(args.batch_size):
            try:
                ex = next(ds_iter)
            except StopIteration:
                break
            content = ex.get("messages")
            if isinstance(content, list):
                content = " ".join(m.get("content", "") for m in content if isinstance(m, dict))
            elif not isinstance(content, str):
                content = str(content or "")
            if content.strip():
                texts.append(content)
        if not texts:
            break

        enc = tok(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_tokens,
            padding=True,
        )
        ids = enc["input_ids"].to("cuda")
        attn = enc["attention_mask"].to("cuda")

        with torch.no_grad():
            captured.clear()
            _ = model(input_ids=ids, attention_mask=attn)

        # captured["residual"] / ["intermediate"] are GPU tensors; pull each
        # row at its real seq_len and quantize to int8 immediately so we
        # free GPU memory before the next batch.
        for b in range(ids.shape[0]):
            real_len = int(attn[b].sum().item())
            tokens = ids[b, :real_len].cpu().numpy().astype(np.int32)
            sample = {"tokens": tokens.tobytes(), "n_tokens": int(real_len)}
            for key in ("residual", "intermediate"):
                if key not in captured:
                    continue
                arr = captured[key][b, :real_len, :].float().cpu().numpy()
                q, scale = quantize_int8(arr)
                sample[f"{key}_q"] = q.tobytes()
                sample[f"{key}_scale"] = scale
                sample[f"{key}_shape"] = list(arr.shape)
            buf.append(sample)
            yielded += 1
            pbar.update(1)

            if len(buf) >= args.shard_size:
                _flush(buf, args.out, shard_idx, pa, pq)
                buf = []
                shard_idx += 1

        # Aggressive cleanup — KEY for staying under VRAM ceiling
        del ids, attn, _
        captured.clear()
        torch.cuda.empty_cache()
        gc.collect()

    if buf:
        _flush(buf, args.out, shard_idx, pa, pq)
        shard_idx += 1

    pbar.close()
    print(f"[corpus] done. {yielded} sequences in {shard_idx} shards "
          f"under {args.out}", flush=True)
    return 0


def _flush(buf: list[dict], out: Path, shard_idx: int, pa, pq) -> None:
    """Write a shard atomically (write to .tmp, then rename)."""
    final = out / f"shard_{shard_idx:04d}.parquet"
    tmp = out / f"shard_{shard_idx:04d}.parquet.tmp"
    table = pa.Table.from_pylist(buf)
    pq.write_table(table, str(tmp), compression="zstd")
    os.replace(tmp, final)


if __name__ == "__main__":
    sys.exit(main())
