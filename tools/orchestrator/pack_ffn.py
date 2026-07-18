#!/usr/bin/env python3
"""pack_ffn — convert dismantle's raw FFN-sparsity capture stream into the
sparsity predictor trainer's int8 parquet shards.

dismantle's `generate` (with HAWKING_QWEN_CAPTURE_FFN_PATH=<file>) appends a
little-endian binary stream. At every transformer layer of every greedy decode
step it records (a) `norm_in` = the `ffn_norm` RMSNorm output (the gate/up
input, i.e. the predictor's input) and (b) per 256-channel block of the
intermediate, `max|silu(gate)*up|` (the active-block label source). Capture runs
on the non-TCB `forward_token` path so these are computed on-host.

Binary stream layout (all little-endian)
----------------------------------------
Per decoded sequence, in order:
  * sentinel : u32 0xFFFFFFFF, u32 0xFFFFFFFF, u32 hidden, u32 n_blocks
  * then per (decode step, layer):
        u32 layer, f32 norm_in[hidden],
        f32 blockmax[n_blocks], f32 blockl2[n_blocks]
Within a sequence the layers appear once per decode step in layer order, so
grouping records by layer preserves token order per layer.

Output parquet schema (matches colab/sparsity_predictor_train.py § load_layer,
plus an extra act_blockl2 column for the Step-2 sparsity gate)
------------------------------------------------------------------------------
Each row = one (sequence, layer):
  * layer              : int32
  * norm_in_q          : bytes int8  (n_tok * hidden)
  * norm_in_scale      : f32
  * act_blockmax_q     : bytes int8  (n_tok * n_blocks)   max|silu*up| (trainer label)
  * act_blockmax_scale : f32
  * act_blockl2_q      : bytes int8  (n_tok * n_blocks)   ||silu*up||_2 (gate proxy)
  * act_blockl2_scale  : f32
  * shape              : list[int]   [n_tok, hidden, n_blocks]

Usage
-----
    python3.12 tools/orchestrator/pack_ffn.py \
        --in _capture/q3b_ffn.bin \
        --out-dir _capture/q3b_ffn_shards \
        --rows-per-shard 256
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
    sys.stderr.write("pack_ffn needs pyarrow: pip install pyarrow\n")
    sys.exit(1)

SENTINEL = 0xFFFFFFFF


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
    """Yield (hidden, n_blocks, {layer: (norm_in[n,h], blockmax[n,nb])}).

    One dict per captured sequence; each layer's arrays are stacked in decode
    order.
    """
    data = path.read_bytes()
    n = len(data)
    off = 0
    hidden: int | None = None
    n_blocks: int | None = None
    # layer -> ([norm_in rows], [blockmax rows], [blockl2 rows])
    cur: dict[int, tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]] = {}

    def flush():
        if hidden is None or not cur:
            return None
        out = {
            layer: (
                np.stack(norms).astype(np.float32),
                np.stack(maxes).astype(np.float32),
                np.stack(l2s).astype(np.float32),
            )
            for layer, (norms, maxes, l2s) in cur.items()
        }
        return hidden, n_blocks, out

    while off + 8 <= n:
        a, b = struct.unpack_from("<II", data, off)
        if a == SENTINEL and b == SENTINEL:
            if off + 16 > n:
                break  # truncated sentinel
            _, _, h, nb = struct.unpack_from("<IIII", data, off)
            seq = flush()
            if seq is not None:
                yield seq
            hidden, n_blocks = int(h), int(nb)
            cur = {}
            off += 16
            continue
        if hidden is None or n_blocks is None:
            raise ValueError(f"ffn stream did not start with a sentinel at off={off}")
        rec_bytes = 4 + hidden * 4 + 2 * n_blocks * 4
        if off + rec_bytes > n:
            break  # truncated trailing record (aborted run)
        (layer,) = struct.unpack_from("<I", data, off)
        foff = off + 4
        norm = np.frombuffer(data, dtype=np.float32, count=hidden, offset=foff).copy()
        bmax = np.frombuffer(
            data, dtype=np.float32, count=n_blocks, offset=foff + hidden * 4
        ).copy()
        bl2 = np.frombuffer(
            data, dtype=np.float32, count=n_blocks,
            offset=foff + hidden * 4 + n_blocks * 4,
        ).copy()
        slot = cur.setdefault(int(layer), ([], [], []))
        slot[0].append(norm)
        slot[1].append(bmax)
        slot[2].append(bl2)
        off += rec_bytes

    seq = flush()
    if seq is not None:
        yield seq


def build_rows(hidden: int, n_blocks: int, layers: dict) -> list[dict]:
    rows = []
    for layer, (norm, bmax, bl2) in sorted(layers.items()):
        n_tok = int(norm.shape[0])
        norm_q, norm_scale = quantize_int8(norm)
        bmax_q, bmax_scale = quantize_int8(bmax)
        bl2_q, bl2_scale = quantize_int8(bl2)
        rows.append(
            {
                "layer": np.int32(layer),
                "norm_in_q": norm_q,
                "norm_in_scale": float(norm_scale),
                "act_blockmax_q": bmax_q,
                "act_blockmax_scale": float(bmax_scale),
                "act_blockl2_q": bl2_q,
                "act_blockl2_scale": float(bl2_scale),
                "shape": [n_tok, int(hidden), int(n_blocks)],
            }
        )
    return rows


SCHEMA = pa.schema([
    ("layer", pa.int32()),
    ("norm_in_q", pa.binary()),
    ("norm_in_scale", pa.float32()),
    ("act_blockmax_q", pa.binary()),
    ("act_blockmax_scale", pa.float32()),
    ("act_blockl2_q", pa.binary()),
    ("act_blockl2_scale", pa.float32()),
    ("shape", pa.list_(pa.int64())),
])


def write_shard(rows: list[dict], out_dir: Path, idx: int, schema=SCHEMA) -> Path:
    cols = {k: [r[k] for r in rows] for k in schema.names}
    table = pa.table(cols, schema=schema)
    out = out_dir / f"shard_{idx:05d}.parquet"
    pq.write_table(table, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="pack_ffn")
    ap.add_argument("--in", dest="inp", required=True, help="raw capture .bin")
    ap.add_argument("--out-dir", required=True, help="dir for shard_*.parquet")
    ap.add_argument("--rows-per-shard", type=int, default=256)
    args = ap.parse_args()

    inp = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    shard_idx = 0
    n_seq = n_rows = n_tok_total = 0
    hidden_seen: set[int] = set()
    nblk_seen: set[int] = set()

    for hidden, n_blocks, layers in iter_sequences(inp):
        n_seq += 1
        hidden_seen.add(hidden)
        nblk_seen.add(n_blocks)
        seq_rows = build_rows(hidden, n_blocks, layers)
        for r in seq_rows:
            n_tok_total += int(r["shape"][0])
        rows.extend(seq_rows)
        n_rows += len(seq_rows)
        while len(rows) >= args.rows_per_shard:
            p = write_shard(rows[: args.rows_per_shard], out_dir, shard_idx)
            print(f"[pack-ffn] wrote {p} ({args.rows_per_shard} rows)")
            rows = rows[args.rows_per_shard :]
            shard_idx += 1

    if rows:
        p = write_shard(rows, out_dir, shard_idx)
        print(f"[pack-ffn] wrote {p} ({len(rows)} rows)")
        shard_idx += 1

    print(
        f"[pack-ffn] done: {n_seq} sequences, {n_rows} layer-rows, "
        f"{n_tok_total} layer-token samples, {shard_idx} shards, "
        f"hidden={sorted(hidden_seen)}, n_blocks={sorted(nblk_seen)}"
    )
    if n_rows == 0:
        sys.stderr.write("[pack-ffn] WARNING: no rows emitted — check capture flag\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
