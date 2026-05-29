#!/usr/bin/env python3
"""pack_corpus — convert dismantle's raw quantized-residual capture stream
into the Eagle5 trainer's compact parquet shards.

dismantle's `generate` (with DISMANTLE_QWEN_EAGLE5_CAPTURE=1 and
DISMANTLE_QWEN_CAPTURE_CORPUS_PATH=<file>) appends a little-endian binary
stream of the residuals the *quantized* (Q4_K_M) runtime actually serves.
This is the key fix for the fp16→Q4_K_M distribution shift: the head trains
on the same residuals it will see at inference instead of the fp16 capture
the cloud `mega_calibrate` produced.

Binary stream layout (all little-endian)
----------------------------------------
Per decoded sequence, in order:
  * sentinel       : u32 0xFFFFFFFF, u32 0xFFFFFFFF, u32 hidden
  * then per step  : u32 prev_token, u32 next_token,
                     f32 residual[hidden], f32 intermediate[hidden]

`residual[i]` is the layer-(n-k) residual produced by the forward that
consumed `prev_token`; `next_token` is the greedy token that forward
produced. Because decode is greedy, next_token[i] == prev_token[i+1], so a
sequence's token list is simply [prev_token for each step] and the training
target for residual[i] is token[i+1] — the model's REAL next token.

Output parquet schema (matches eagle5_train_pytorch.py § Data contract)
-----------------------------------------------------------------------
Each row = one sequence:
  * tokens             : bytes  (int32 packed, len n_tok)
  * residual_q         : bytes  (int8  packed, n_tok*hidden)
  * residual_scale     : f32    (single global scale)
  * residual_shape     : list[int]  [n_tok, hidden]
  * intermediate_q / intermediate_scale / intermediate_shape : same

Usage
-----
    python3 tools/orchestrator/pack_corpus.py \
        --in _capture/q7b_residuals.bin \
        --out-dir _capture/q7b_shards \
        --rows-per-shard 64

Then point the trainer at --corpus-dir _capture/q7b_shards.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    sys.stderr.write("pack_corpus needs pyarrow: pip install pyarrow\n")
    sys.exit(1)

SENTINEL = 0xFFFFFFFF
MIN_TOKENS = 5  # trainer drops sequences shorter than this


def quantize_int8(arr: np.ndarray) -> tuple[bytes, float]:
    """Symmetric per-tensor int8 quant with a single global scale.

    Mirrors the trainer's dequant: f32 = int8 * scale.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    amax = float(np.abs(arr).max()) if arr.size else 0.0
    scale = (amax / 127.0) if amax > 0 else 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q.tobytes(), scale


def iter_sequences(path: Path):
    """Yield (tokens int32[n], residual f32[n,h], intermediate f32[n,h])."""
    data = path.read_bytes()
    n = len(data)
    off = 0
    hidden = None
    cur_tokens: list[int] = []
    cur_res: list[np.ndarray] = []
    cur_int: list[np.ndarray] = []

    def flush():
        if hidden is None or len(cur_tokens) < MIN_TOKENS:
            return None
        toks = np.asarray(cur_tokens, dtype=np.int32)
        res = np.stack(cur_res).astype(np.float32)
        inter = np.stack(cur_int).astype(np.float32)
        return toks, res, inter

    while off + 12 <= n:
        a, b, h = struct.unpack_from("<III", data, off)
        if a == SENTINEL and b == SENTINEL:
            # Sequence boundary: emit the one we were building, start fresh.
            seq = flush()
            if seq is not None:
                yield seq
            hidden = h
            cur_tokens, cur_res, cur_int = [], [], []
            off += 12
            continue
        if hidden is None:
            # Stream must start with a sentinel.
            raise ValueError(f"corpus stream did not start with a sentinel at off={off}")
        step_bytes = 8 + hidden * 4 * 2
        if off + step_bytes > n:
            break  # truncated trailing record (e.g. aborted run)
        prev_id, _next_id = struct.unpack_from("<II", data, off)
        foff = off + 8
        res = np.frombuffer(data, dtype=np.float32, count=hidden, offset=foff).copy()
        inter = np.frombuffer(
            data, dtype=np.float32, count=hidden, offset=foff + hidden * 4
        ).copy()
        cur_tokens.append(prev_id)
        cur_res.append(res)
        cur_int.append(inter)
        off += step_bytes

    seq = flush()
    if seq is not None:
        yield seq


def build_row(toks: np.ndarray, res: np.ndarray, inter: np.ndarray) -> dict:
    res_q, res_scale = quantize_int8(res)
    int_q, int_scale = quantize_int8(inter)
    return {
        "tokens": toks.tobytes(),
        "residual_q": res_q,
        "residual_scale": float(res_scale),
        "residual_shape": [int(res.shape[0]), int(res.shape[1])],
        "intermediate_q": int_q,
        "intermediate_scale": float(int_scale),
        "intermediate_shape": [int(inter.shape[0]), int(inter.shape[1])],
    }


SCHEMA = pa.schema([
    ("tokens", pa.binary()),
    ("residual_q", pa.binary()),
    ("residual_scale", pa.float32()),
    ("residual_shape", pa.list_(pa.int64())),
    ("intermediate_q", pa.binary()),
    ("intermediate_scale", pa.float32()),
    ("intermediate_shape", pa.list_(pa.int64())),
])


def write_shard(rows: list[dict], out_dir: Path, idx: int) -> Path:
    cols = {k: [r[k] for r in rows] for k in SCHEMA.names}
    table = pa.table(cols, schema=SCHEMA)
    out = out_dir / f"shard_{idx:05d}.parquet"
    pq.write_table(table, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="pack_corpus")
    ap.add_argument("--in", dest="inp", required=True, help="raw capture .bin")
    ap.add_argument("--out-dir", required=True, help="dir for shard_*.parquet")
    ap.add_argument("--rows-per-shard", type=int, default=64)
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    shard_idx = 0
    n_seq = n_steps = n_dropped = 0
    hidden_seen = set()

    for toks, res, inter in iter_sequences(inp):
        if len(toks) < MIN_TOKENS:
            n_dropped += 1
            continue
        n_seq += 1
        n_steps += len(toks)
        hidden_seen.add(res.shape[1])
        rows.append(build_row(toks, res, inter))
        if len(rows) >= args.rows_per_shard:
            p = write_shard(rows, out_dir, shard_idx)
            print(f"[pack] wrote {p} ({len(rows)} rows)")
            rows = []
            shard_idx += 1

    if rows:
        p = write_shard(rows, out_dir, shard_idx)
        print(f"[pack] wrote {p} ({len(rows)} rows)")
        shard_idx += 1

    print(f"[pack] done: {n_seq} sequences, {n_steps} steps, "
          f"{shard_idx} shards, hidden={sorted(hidden_seen)}, dropped={n_dropped}")
    if n_seq == 0:
        sys.stderr.write("[pack] WARNING: no sequences emitted — check capture flags\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
