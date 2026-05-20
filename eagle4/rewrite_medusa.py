"""Phase F.1 — window-join rewrite of eagle4_v0 shards into eagle4_v0_medusa.

Adds K-1 next-position columns to each row, derived from the same-sample
row at position+i within the existing shards. No GPU, no model load —
pure parquet rewrite.

For each row at (sample_id=S, pos=p), adds for i in 1..K-1:
    next_token_p{i}   = row(S, p+i).next_token
    hidden_high_p{i}  = row(S, p+i).hidden_high

Tail rows whose +i sibling falls past end-of-sample get sentinels:
    next_token_p{i}   = -1
    hidden_high_p{i}  = 4096 zero bytes
F.2 dataloader uses `next_token_p{i} != -1` as the per-head loss mask.

Streaming pass over shards. Rows within a shard are in (sample_id,
position) order by capture.py's construction, so we accumulate one
sample at a time and emit when sample_id transitions. Cross-shard
samples are handled because state crosses shard boundaries.

Output row count == input row count; shards re-tiled at SHARD_ROWS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

K = 8
HIDDEN = 2048
H_BYTES = HIDDEN * 2  # fp16
ZERO_HIDDEN = bytes(H_BYTES)
SHARD_ROWS = 8192

BASE_FIELDS = [
    ("sample_id", pa.string()),
    ("position", pa.int32()),
    ("prev_token", pa.int32()),
    ("next_token", pa.int32()),
    ("hidden_low", pa.binary()),
    ("hidden_mid", pa.binary()),
    ("hidden_high", pa.binary()),
    ("shared_hidden", pa.binary()),
    ("router_logits_per_layer", pa.binary()),
    ("routed_mask_per_layer", pa.binary()),
]
NEW_FIELDS = (
    BASE_FIELDS
    + [(f"next_token_p{i}", pa.int32()) for i in range(1, K)]
    + [(f"hidden_high_p{i}", pa.binary()) for i in range(1, K)]
)
NEW_SCHEMA = pa.schema(NEW_FIELDS)


def rewrite(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(in_dir.glob("shard_*.parquet"))
    if not shards:
        print(f"[rewrite] no shards in {in_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[rewrite] {len(shards)} input shards", flush=True)

    pending_sid: str | None = None
    pending: list[dict] = []
    out_buf: list[dict] = []
    out_shard = 0
    rows_in = 0
    rows_out = 0
    tail_drops_per_i = [0] * K
    t0 = time.time()

    def flush_out() -> None:
        nonlocal out_buf, out_shard, rows_out
        while len(out_buf) >= SHARD_ROWS:
            path = out_dir / f"shard_{out_shard:05d}.parquet"
            tbl = pa.Table.from_pylist(out_buf[:SHARD_ROWS], schema=NEW_SCHEMA)
            pq.write_table(tbl, path, compression="zstd")
            rows_out += SHARD_ROWS
            out_buf = out_buf[SHARD_ROWS:]
            out_shard += 1

    def emit_pending() -> None:
        n = len(pending)
        for idx, row in enumerate(pending):
            new_row = dict(row)
            for i in range(1, K):
                j = idx + i
                if j < n:
                    new_row[f"next_token_p{i}"] = int(pending[j]["next_token"])
                    new_row[f"hidden_high_p{i}"] = pending[j]["hidden_high"]
                else:
                    new_row[f"next_token_p{i}"] = -1
                    new_row[f"hidden_high_p{i}"] = ZERO_HIDDEN
                    tail_drops_per_i[i] += 1
            out_buf.append(new_row)
        flush_out()

    for sp_idx, shard_path in enumerate(shards):
        table = pq.read_table(shard_path)
        rows = table.to_pylist()
        rows_in += len(rows)
        for row in rows:
            sid = row["sample_id"]
            if sid != pending_sid:
                if pending_sid is not None:
                    emit_pending()
                    pending = []
                pending_sid = sid
            pending.append(row)
        elapsed = time.time() - t0
        rate = rows_in / max(elapsed, 1e-3)
        print(
            f"[rewrite] read {sp_idx+1}/{len(shards)} ({rows_in} rows, "
            f"emit {rows_out}, buf {len(out_buf)}, {rate:.0f} r/s)",
            flush=True,
        )

    if pending:
        emit_pending()
    if out_buf:
        path = out_dir / f"shard_{out_shard:05d}.parquet"
        tbl = pa.Table.from_pylist(out_buf, schema=NEW_SCHEMA)
        pq.write_table(tbl, path, compression="zstd")
        rows_out += len(out_buf)
        out_shard += 1

    elapsed = time.time() - t0
    print(
        f"[rewrite] done: {rows_out} rows in {out_shard} shards "
        f"({elapsed:.1f}s, {rows_out/max(elapsed,1):.0f} r/s)",
        flush=True,
    )
    print(f"[rewrite] rows_in={rows_in} rows_out={rows_out} (must match)", flush=True)
    assert rows_in == rows_out, f"row count mismatch: {rows_in} != {rows_out}"
    print(
        "[rewrite] tail-sentinel counts per i: "
        + ", ".join(f"p{i}={tail_drops_per_i[i]}" for i in range(1, K)),
        flush=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(prog="rewrite-medusa")
    p.add_argument(
        "--in-dir",
        type=Path,
        default=Path("../training_data/c2_hidden/eagle4_v0"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("../training_data/c2_hidden/eagle4_v0_medusa"),
    )
    args = p.parse_args()
    rewrite(args.in_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
