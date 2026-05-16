#!/usr/bin/env python3
"""
pre_shuffle.py — one-shot offline shuffler for DCAP shards.

Fixes the streaming iterator's throughput cliff (known issue #2 in
README) by replacing online shuffle-buffer thrashing with a single
deterministic offline shuffle. After running this, training does pure
sequential reads from the shuffled output and the iterator can drop
its shuffle-buffer logic entirely.

Strategy:
  1. Index pass: scan input shard, record (offset, length) per record.
  2. Shuffle the (offset, length) list with a fixed seed.
  3. Sequential output: write records in shuffled order to a new .bin
     using random access on the input file.

Memory cost: ~24 bytes/record × N_records. For 55K samples × ~100 records/
sample = 5.5M records → ~132 MB index. Fits comfortably.
Disk cost: 2x shard size during shuffle (input + output coexist).

After this, training command becomes:
  $PY tools/training/mlx_eagle/train.py \\
    --parquet none \\
    --raw-bin training_data/c2_hidden/eagle3_v0/shard_000.shuffled.bin \\
    ...
(--raw-bin is a future-work; for now use this script + then re-convert
the shuffled .bin to parquet for the existing iterator. Or just trust
that the parquet shuffle is fine — pre-shuffle is the belt-and-suspenders
fix when the streaming cliff really bites.)
"""

from __future__ import annotations
import argparse
import pathlib
import struct
import sys
import random
import time


HEADER_SIZE = 16
MAGIC = b"DCAP"


def index_records(shard_path: pathlib.Path, hidden_dim: int):
    """Scan shard, yield (offset, length) for each record."""
    hb_bytes = hidden_dim * 2
    with open(shard_path, "rb") as f:
        hdr = f.read(HEADER_SIZE)
        if hdr[:4] != MAGIC:
            raise ValueError(f"bad magic in {shard_path}")
        hd = struct.unpack("<I", hdr[8:12])[0]
        if hd != hidden_dim:
            raise ValueError(f"shard hidden_dim={hd} != expected {hidden_dim}")
        offset = HEADER_SIZE
        while True:
            f.seek(offset)
            lb = f.read(2)
            if not lb or len(lb) < 2:
                return
            (id_len,) = struct.unpack("<H", lb)
            rec_len = 2 + id_len + 12 + hb_bytes
            yield offset, rec_len
            offset += rec_len


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--hidden-dim", type=int, default=2048)
    args = p.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)

    t0 = time.time()
    print(f"[pre_shuffle] indexing {src}", file=sys.stderr)
    index = list(index_records(src, args.hidden_dim))
    n = len(index)
    print(f"[pre_shuffle] {n:,} records indexed in {time.time()-t0:.1f}s "
          f"(~{24*n/1024/1024:.0f} MB index)", file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(index)

    t1 = time.time()
    print(f"[pre_shuffle] writing shuffled output to {dst}", file=sys.stderr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Open src for random access, dst for sequential append.
    src_f = open(src, "rb")
    # Read + reuse the header.
    src_f.seek(0)
    header = src_f.read(HEADER_SIZE)
    with open(dst, "wb") as out:
        out.write(header)
        for i, (off, rec_len) in enumerate(index):
            src_f.seek(off)
            out.write(src_f.read(rec_len))
            if (i + 1) % 100000 == 0:
                print(f"  wrote {i+1:,}/{n:,}", file=sys.stderr)
    src_f.close()
    elapsed = time.time() - t1
    print(f"[pre_shuffle] done. wrote {dst.stat().st_size:,} bytes in {elapsed:.1f}s "
          f"({n/elapsed:.0f} records/s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
