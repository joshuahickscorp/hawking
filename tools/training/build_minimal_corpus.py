#!/usr/bin/env python3
"""Build a slim derived corpus that keeps only what eagle5_train.py reads.

Reads `artifacts/calibration/v2_lite_corpus/shard_*.parquet` and writes
`artifacts/calibration/v2_lite_corpus_min/shard_*.parquet` with the same
filename mapping, same row counts, same `tokens`/`n_tokens` columns, and
the heavyweight per-layer arrays trimmed to just the capture layer.

Disk math (V2-Lite, capture_layer=25):
  Original shard ≈ 870 MB (27 layers × residual + 26 MoE layers × intermediate)
  Minimal shard ≈ 45 MB  (only layer-25 residual + MoE-layer-24 intermediate)
  → ~95% reduction, ~4.5 GB minimal corpus vs 80 GB original.

The output column shapes mirror the originals so eagle5_train.py
(and eagle5_tau_eval.py) work unchanged: indices outside the capture
layer are empty lists, which the trainer never reads.

Usage:
  python3 tools/training/build_minimal_corpus.py
  python3 tools/training/build_minimal_corpus.py --capture-layer 25 --n-moe-first-dense 1

Idempotent: skips shards whose output already exists.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


N_LAYERS = 27       # V2-Lite total transformer blocks
N_MOE_LAYERS = 26   # MoE-bearing blocks (layer 0 is dense)


def extract_one_shard(src: Path, dst: Path, capture_layers: list[int], n_moe_first_dense: int) -> tuple[int, int, int]:
    """Returns (rows, src_bytes, dst_bytes).

    capture_layers: list of V2-Lite block indices to KEEP. Others stored
    as empty lists. The corresponding MoE intermediates (layer-N - n_dense)
    are auto-derived.
    """
    moe_idxs = [li - n_moe_first_dense for li in capture_layers
                if li - n_moe_first_dense >= 0]
    keep_layers = set(capture_layers)
    keep_moe = set(moe_idxs)

    tbl = pq.read_table(src)
    rows = tbl.num_rows

    out_cols: dict[str, list] = {
        "tokens": tbl["tokens"].to_pylist(),
        "n_tokens": tbl["n_tokens"].to_pylist(),
    }

    # Extract residual_in_per_layer for ALL capture_layers, pad others.
    if "residual_in_per_layer" in tbl.column_names:
        res_col = tbl["residual_in_per_layer"].to_pylist()
        slim_res = []
        for row in res_col:
            if row is None:
                slim_res.append([[] for _ in range(N_LAYERS)])
                continue
            padded = [[] for _ in range(N_LAYERS)]
            for li in keep_layers:
                if li < len(row):
                    padded[li] = row[li]
            slim_res.append(padded)
        out_cols["residual_in_per_layer"] = slim_res

    # Extract intermediate_per_layer for matching MoE indices, pad others.
    if "intermediate_per_layer" in tbl.column_names:
        inter_col = tbl["intermediate_per_layer"].to_pylist()
        slim_inter = []
        for row in inter_col:
            if row is None:
                slim_inter.append([{"layer": -1, "raw": []} for _ in range(N_MOE_LAYERS)])
                continue
            padded = [{"layer": -1, "raw": []} for _ in range(N_MOE_LAYERS)]
            for moe_idx in keep_moe:
                if moe_idx < len(row):
                    padded[moe_idx] = row[moe_idx]
            slim_inter.append(padded)
        out_cols["intermediate_per_layer"] = slim_inter

    # Keep the small routing columns — they're useful for future levers and
    # cost almost nothing on disk.
    for col_name in ("expert_idx_per_layer", "routing_topk_weight_per_layer"):
        if col_name in tbl.column_names:
            out_cols[col_name] = tbl[col_name].to_pylist()

    # Write.
    out_tbl = pa.Table.from_pydict(out_cols)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, dst, compression="zstd")

    return rows, src.stat().st_size, dst.stat().st_size


def _worker(args_tuple):
    """ProcessPool worker — does one shard end-to-end."""
    src, dst, capture_layers, n_moe_first_dense = args_tuple
    if dst.exists():
        sz = dst.stat().st_size
        return (str(src.name), 0, src.stat().st_size, sz, "skipped")
    try:
        rows, sb, db = extract_one_shard(src, dst, capture_layers, n_moe_first_dense)
        return (str(src.name), rows, sb, db, "ok")
    except Exception as e:  # noqa: BLE001
        return (str(src.name), 0, 0, 0, f"error: {e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", type=Path,
                   default=Path("artifacts/calibration/v2_lite_corpus"))
    p.add_argument("--dst-dir", type=Path,
                   default=Path("artifacts/calibration/v2_lite_corpus_min"))
    p.add_argument("--capture-layers", type=str, default="13,25",
                   help="comma-separated V2-Lite block indices whose residual + "
                        "intermediate to keep. Default 13,25 covers eagle5 v2 "
                        "(layer 25) AND eagle5 v3 alt-config (layer 13).")
    p.add_argument("--n-moe-first-dense", type=int, default=1,
                   help="number of dense (non-MoE) layers before the first MoE block; "
                        "V2-Lite has 1 dense layer at index 0")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="skip output shards that already exist")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                   help="parallel shard extractors (defaults to CPU count - 2)")
    args = p.parse_args()

    shards = sorted(args.src_dir.glob("shard_*.parquet"))
    if not shards:
        print(f"no shards in {args.src_dir}", file=sys.stderr)
        return 2

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    total_src = 0
    total_dst = 0
    total_rows = 0
    completed = 0
    t0 = time.time()

    print(f"extracting {len(shards)} shards with {args.workers} workers…",
          file=sys.stderr)

    capture_layers = [int(x) for x in args.capture_layers.split(",") if x.strip()]
    print(f"keeping layers {capture_layers} + MoE indices "
          f"{[x - args.n_moe_first_dense for x in capture_layers if x >= args.n_moe_first_dense]}…",
          file=sys.stderr)
    work = [(s, args.dst_dir / s.name, capture_layers, args.n_moe_first_dense)
            for s in shards]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, rows, sb, db, status in ex.map(_worker, work):
            completed += 1
            total_src += sb
            total_dst += db
            total_rows += rows
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(shards) - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{len(shards)}] {name}: {rows} rows, "
                  f"{sb/1e6:.0f}MB → {db/1e6:.1f}MB ({status}, "
                  f"elapsed={elapsed:.0f}s, eta={eta:.0f}s)",
                  file=sys.stderr)

    total_min = max(time.time() - t0, 1) / 60
    print(f"\ndone in {total_min:.1f} min. {len(shards)} shards.",
          file=sys.stderr)
    print(f"  source size:  {total_src/1e9:.1f} GB",
          file=sys.stderr)
    print(f"  minimal size: {total_dst/1e9:.2f} GB  "
          f"({100*total_dst/total_src:.1f}% of original)",
          file=sys.stderr)
    print(f"\nIf eagle5_train + tau_eval succeed against the new dir, you can "
          f"safely `rm -rf {args.src_dir}` to reclaim ~{(total_src-total_dst)/1e9:.0f} GB.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
