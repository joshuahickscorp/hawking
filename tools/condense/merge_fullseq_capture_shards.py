#!/usr/bin/env python3
"""Merge disjoint DSV4F fullseq capture worker directories into one coherent set.

Each worker must have captured a disjoint sequence shard into its own --out-dir
(never shared files). This script:

  * copies/links trace JSON shards into out/traces/
  * concatenates activations/LXX.npy rows in global seq_index / example_id order
  * writes example_ids.json + per-layer export meta
  * seals DSV4F_FULLSEQ_CAPTURE_MERGED_RECEIPT.json with per-worker provenance

Correctness bar for multi-worker: bit-exact npy/hash match vs a single serial
run of the same sequence set (see FULLSEQ_CAPTURE_PARALLELISM_FINDINGS.json).

Usage:
  python3 tools/condense/merge_fullseq_capture_shards.py \\
    --out /path/to/merged \\
    --worker /path/to/w0 \\
    --worker /path/to/w1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any


RECEIPT_NAME = "DSV4F_FULLSEQ_CAPTURE_RECEIPT.json"
MERGED_RECEIPT_NAME = "DSV4F_FULLSEQ_CAPTURE_MERGED_RECEIPT.json"
MERGED_SCHEMA = "hawking.gravity.deepseek_v4.fullseq_capture_merged.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_npy_f32_2d(path: Path) -> tuple[list[int], bytes]:
    """Minimal numpy .npy (v1/v2) reader for C-order float32 2-D arrays."""
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError(f"{path}: not a .npy file")
    major, minor = raw[6], raw[7]
    if major == 1:
        header_len = struct.unpack_from("<H", raw, 8)[0]
        header_start = 10
    elif major == 2:
        header_len = struct.unpack_from("<I", raw, 8)[0]
        header_start = 12
    else:
        raise ValueError(f"{path}: unsupported npy version {major}.{minor}")
    header = raw[header_start : header_start + header_len].decode("latin1")
    # header is a Python dict literal like: {'descr': '<f4', 'fortran_order': False, 'shape': (N, D), }
    meta = eval(header, {"__builtins__": {}}, {})  # noqa: S307 — controlled npy header
    if meta.get("fortran_order"):
        raise ValueError(f"{path}: fortran_order not supported")
    descr = meta["descr"]
    if descr not in ("<f4", "|f4", "float32"):
        raise ValueError(f"{path}: expected float32 descr, got {descr!r}")
    shape = list(meta["shape"])
    if len(shape) != 2:
        raise ValueError(f"{path}: expected 2-D shape, got {shape}")
    data = raw[header_start + header_len :]
    expected = shape[0] * shape[1] * 4
    if len(data) < expected:
        raise ValueError(f"{path}: truncated payload {len(data)} < {expected}")
    return shape, data[:expected]


def write_npy_f32_2d(path: Path, data: bytes, n: int, d: int) -> None:
    """Write C-order float32 2-D .npy (v1 header)."""
    header_dict = (
        f"{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {d}), }}"
    )
    # pad to 64-byte alignment: 10 + len(header) + 1 (newline) ≡ 0 (mod 64)
    pad = 64 - ((10 + len(header_dict) + 1) % 64)
    if pad == 64:
        pad = 0
    header = header_dict + (" " * pad) + "\n"
    header_bytes = header.encode("latin1")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"\x93NUMPY")
        f.write(bytes([1, 0]))
        f.write(struct.pack("<H", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


def load_worker(path: Path) -> dict[str, Any]:
    receipt_path = path / RECEIPT_NAME
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    shard = receipt.get("shard") or {}
    seq_start = int(shard.get("seq_start", 0))
    seq_end = shard.get("seq_end")
    if seq_end is None:
        # Unsharded full worker: treat as [0, n)
        seq_end = int(receipt.get("scope", {}).get("sequences", 0))
    seq_end = int(seq_end)
    worker_id = shard.get("worker_id") or path.name
    traces_dir = path / "traces"
    trace_files = sorted(traces_dir.glob("*.json")) if traces_dir.is_dir() else []
    act_dir = path / "activations"
    example_ids: list[str] = []
    ids_path = act_dir / "example_ids.json"
    if ids_path.is_file():
        ids_doc = json.loads(ids_path.read_text())
        example_ids = list(ids_doc.get("example_ids") or [])
    npy_by_layer: dict[int, Path] = {}
    if act_dir.is_dir():
        for p in act_dir.glob("L*.npy"):
            # L00.npy
            stem = p.stem  # L00
            if len(stem) >= 2 and stem[0] == "L" and stem[1:].isdigit():
                npy_by_layer[int(stem[1:])] = p
    return {
        "path": path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": sha256_file(receipt_path),
        "seq_start": seq_start,
        "seq_end": seq_end,
        "worker_id": worker_id,
        "trace_files": trace_files,
        "example_ids": example_ids,
        "npy_by_layer": npy_by_layer,
    }


def ranges_disjoint(workers: list[dict[str, Any]]) -> None:
    ranges = sorted((w["seq_start"], w["seq_end"], w["worker_id"]) for w in workers)
    for i in range(len(ranges) - 1):
        a0, a1, aid = ranges[i]
        b0, b1, bid = ranges[i + 1]
        if a1 > b0:
            raise SystemExit(
                f"overlapping shards: worker {aid} [{a0},{a1}) overlaps "
                f"worker {bid} [{b0},{b1})"
            )


def merge(workers: list[dict[str, Any]], out: Path, copy_mode: str) -> dict[str, Any]:
    ranges_disjoint(workers)
    workers_sorted = sorted(workers, key=lambda w: w["seq_start"])

    out.mkdir(parents=True, exist_ok=True)
    traces_out = out / "traces"
    traces_out.mkdir(parents=True, exist_ok=True)
    act_out = out / "activations"
    act_out.mkdir(parents=True, exist_ok=True)

    # Traces: copy/link, detect duplicate example_ids
    seen_ids: set[str] = set()
    merged_trace_rows: list[dict[str, Any]] = []
    for w in workers_sorted:
        for tp in w["trace_files"]:
            eid = tp.stem
            if eid in seen_ids:
                raise SystemExit(f"duplicate example_id {eid} across workers")
            seen_ids.add(eid)
            dest = traces_out / tp.name
            if dest.exists():
                raise SystemExit(f"trace destination already exists: {dest}")
            if copy_mode == "hardlink":
                os.link(tp, dest)
            elif copy_mode == "symlink":
                os.symlink(tp.resolve(), dest)
            else:
                shutil.copy2(tp, dest)
            merged_trace_rows.append(
                {
                    "example_id": eid,
                    "path": str(dest),
                    "source_worker_id": w["worker_id"],
                    "source_path": str(tp),
                    "sha256": sha256_file(dest),
                }
            )

    # example_ids + npy concat in worker seq order
    merged_ids: list[str] = []
    for w in workers_sorted:
        if w["example_ids"]:
            merged_ids.extend(w["example_ids"])
        else:
            # Fall back to trace stems in filename order (not ideal; warn)
            for tp in w["trace_files"]:
                merged_ids.append(tp.stem)

    ids_doc = {
        "schema": "hawking.gravity.deepseek_v4.fullseq_activation_export_ids.v1",
        "example_ids": merged_ids,
        "n": len(merged_ids),
        "mode": "offline_analysis_diagnostic",
        "host_activation_handoff_permitted": False,
        "merged_from_workers": [w["worker_id"] for w in workers_sorted],
    }
    ids_path = act_out / "example_ids.json"
    ids_path.write_text(json.dumps(ids_doc, indent=2) + "\n")

    all_layers: set[int] = set()
    for w in workers_sorted:
        all_layers.update(w["npy_by_layer"].keys())

    sidecars: list[dict[str, Any]] = []
    for layer in sorted(all_layers):
        chunks: list[bytes] = []
        total_n = 0
        d: int | None = None
        for w in workers_sorted:
            p = w["npy_by_layer"].get(layer)
            if p is None:
                # Worker captured no export for this layer (hashes-only shard)
                continue
            shape, data = read_npy_f32_2d(p)
            n_i, d_i = shape
            if d is None:
                d = d_i
            elif d != d_i:
                raise SystemExit(
                    f"layer L{layer:02d}: width mismatch {d} vs {d_i} from {p}"
                )
            chunks.append(data)
            total_n += n_i
        if d is None or total_n == 0:
            continue
        flat = b"".join(chunks)
        npy_path = act_out / f"L{layer:02d}.npy"
        write_npy_f32_2d(npy_path, flat, total_n, d)
        meta_path = act_out / f"L{layer:02d}.export.json"
        meta = {
            "schema": "hawking.gravity.deepseek_v4.fullseq_activation_export.v1",
            "layer": layer,
            "site": "late_hidden",
            "npy": str(npy_path),
            "shape": [total_n, d],
            "dtype": "float32",
            "example_ids_path": str(ids_path),
            "mode": "offline_analysis_diagnostic",
            "host_activation_handoff_permitted": False,
            "merged": True,
            "npy_sha256": sha256_file(npy_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        sidecars.append(
            {
                "layer": layer,
                "path": str(npy_path),
                "shape": [total_n, d],
                "dtype": "float32",
                "site": "late_hidden",
                "npy_sha256": meta["npy_sha256"],
            }
        )

    # Aggregate metal / wall from workers
    metal_dispatches = 0
    command_buffers = 0
    cpu_visible_waits = 0
    tokens_total = 0
    wall_ms_sum = 0.0
    wall_ms_max = 0.0
    layers_run: list[int] | None = None
    for w in workers_sorted:
        r = w["receipt"]
        m = r.get("metal") or {}
        metal_dispatches += int(m.get("metal_dispatches") or 0)
        command_buffers += int(m.get("command_buffers") or 0)
        cpu_visible_waits += int(m.get("cpu_visible_waits") or 0)
        tokens_total += int((r.get("scope") or {}).get("tokens_total") or 0)
        wt = float(r.get("wall_time_ms") or 0.0)
        wall_ms_sum += wt
        wall_ms_max = max(wall_ms_max, wt)
        lr = (r.get("scope") or {}).get("layers_run")
        if isinstance(lr, list):
            if layers_run is None:
                layers_run = list(lr)
            elif layers_run != lr:
                # Allow union if workers used same stack; else record both
                pass

    merged = {
        "schema": MERGED_SCHEMA,
        "status": f"PASS_FULLSEQ_CAPTURE_MERGED_{len(merged_ids)}_SEQS_{len(workers_sorted)}_WORKERS",
        "workers": [
            {
                "worker_id": w["worker_id"],
                "path": str(w["path"]),
                "seq_start": w["seq_start"],
                "seq_end": w["seq_end"],
                "receipt_path": str(w["receipt_path"]),
                "receipt_sha256": w["receipt_sha256"],
                "n_sequences": int(
                    (w["receipt"].get("scope") or {}).get("sequences") or 0
                ),
                "wall_time_ms": w["receipt"].get("wall_time_ms"),
            }
            for w in workers_sorted
        ],
        "scope": {
            "sequences": len(merged_ids),
            "example_ids": merged_ids,
            "tokens_total": tokens_total,
            "layers_run": layers_run,
            "seq_range_union": [
                workers_sorted[0]["seq_start"],
                workers_sorted[-1]["seq_end"],
            ]
            if workers_sorted
            else [0, 0],
        },
        "paired_traces": {
            "n_traces": len(merged_trace_rows),
            "dir": str(traces_out),
            "rows": merged_trace_rows,
        },
        "host_activation_export": {
            "enabled": bool(sidecars),
            "sidecars": sidecars,
            "dir": str(act_out),
            "example_ids_path": str(ids_path),
        },
        "metal_aggregated": {
            "metal_dispatches": metal_dispatches,
            "command_buffers": command_buffers,
            "cpu_visible_waits": cpu_visible_waits,
            "note": "sum across workers (not comparable to single-process counters under contention)",
        },
        "wall_time": {
            "sum_worker_ms": wall_ms_sum,
            "max_worker_ms": wall_ms_max,
            "note": "max_worker_ms approximates parallel wall if workers started together",
        },
        "honesty": {
            "fabricated_activations": False,
            "merge_only_reorders_disjoint_shards": True,
            "layer_shard_supported": False,
        },
    }
    pretty = json.dumps(merged, indent=2) + "\n"
    merged_path = out / MERGED_RECEIPT_NAME
    merged_path.write_text(pretty)
    merged["merged_receipt_path"] = str(merged_path)
    merged["merged_receipt_sha256"] = sha256_bytes(pretty.encode())
    # re-write with seal
    pretty2 = json.dumps(merged, indent=2) + "\n"
    merged_path.write_text(pretty2)
    print(json.dumps({"merged_receipt": str(merged_path), "n_traces": len(merged_trace_rows), "n_ids": len(merged_ids), "layers": [s["layer"] for s in sidecars]}, indent=2))
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="merged output directory")
    ap.add_argument(
        "--worker",
        action="append",
        required=True,
        type=Path,
        help="worker out-dir (repeatable); each must contain DSV4F_FULLSEQ_CAPTURE_RECEIPT.json",
    )
    ap.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="how to place traces into the merged dir (default: copy)",
    )
    args = ap.parse_args(argv)
    workers = [load_worker(p) for p in args.worker]
    merge(workers, args.out, args.copy_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
