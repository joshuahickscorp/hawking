#!/usr/bin/env python3
"""
compute_hidden_stats.py — per-channel mean/std of captured hidden states.

One-shot offline computation. Output is a small .npz (2 * 2048 * 4 = 16 KB)
that train.py loads via `--hidden-stats` and applies as `(h - mean) / std`
at batch construction. This makes Adam's per-channel adaptation less
necessary and typically saves ~10% wall to the same loss.

Streaming, single pass — no need to load the whole shard into RAM.
"""

from __future__ import annotations
import argparse
import pathlib
import struct
import sys
import numpy as np


def compute(shard_path: pathlib.Path, hidden_dim: int = 2048):
    """Welford online accumulator — numerically stable for huge N."""
    n = 0
    mean = np.zeros(hidden_dim, dtype=np.float64)
    m2 = np.zeros(hidden_dim, dtype=np.float64)
    hb_bytes = hidden_dim * 2
    with open(shard_path, "rb") as f:
        hdr = f.read(16)
        if hdr[:4] != b"DCAP":
            raise ValueError(f"bad magic in {shard_path}")
        hd = struct.unpack("<I", hdr[8:12])[0]
        if hd != hidden_dim:
            raise ValueError(f"shard hidden_dim={hd} != expected {hidden_dim}")
        while True:
            lb = f.read(2)
            if not lb:
                break
            (id_len,) = struct.unpack("<H", lb)
            f.seek(id_len + 12, 1)  # skip sample_id + pos + prev + next
            hb = f.read(hb_bytes)
            if len(hb) != hb_bytes:
                break
            x = np.frombuffer(hb, dtype=np.float16).astype(np.float64)
            n += 1
            delta = x - mean
            mean += delta / n
            m2 += delta * (x - mean)
            if n % 50000 == 0:
                print(f"  scanned {n:,} records", file=sys.stderr)
    var = m2 / max(n - 1, 1)
    std = np.sqrt(var)
    # Avoid division by near-zero std on any channel.
    std = np.where(std < 1e-6, 1.0, std)
    return n, mean.astype(np.float32), std.astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--shard", required=True)
    p.add_argument("--hidden-dim", type=int, default=2048)
    p.add_argument("--out", default="tools/training/mlx_eagle/hidden_stats.npz")
    args = p.parse_args()

    print(f"[stats] scanning {args.shard}", file=sys.stderr)
    n, mean, std = compute(pathlib.Path(args.shard), args.hidden_dim)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mean=mean, std=std, n_records=np.int64(n))
    print(
        f"[stats] {n:,} records  mean L2={np.linalg.norm(mean):.3f}  "
        f"std mean={std.mean():.3f}  std range=[{std.min():.4f}, {std.max():.4f}]",
        file=sys.stderr,
    )
    print(f"[stats] wrote {out} ({out.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
