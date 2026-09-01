#!/usr/bin/env python3
"""Measure a routed expert-bank range-read opportunity without changing execution.

The native Flash graph currently reads both complete expert banks before the
Metal dispatch.  This probe uses the already verified route IDs for one exact
layer/state, reads only those expert rows from the pinned safetensors shards,
and records the physical source-I/O delta.  It is a profile, not a runtime
promotion or parity claim; compact-bank kernel integration remains a separate
gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_locations(root: Path, index_name: str) -> dict[str, tuple[Path, int, int, list[int]]]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)
    locations: dict[str, tuple[Path, int, int, list[int]]] = {}
    for shard, names in by_shard.items():
        path = root / shard
        with path.open("rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for name in names:
            info = header[name]
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin, info["shape"])
    return locations


def read_expert_rows(loc, experts: list[int], row_bytes: int, rows_per_expert: int) -> tuple[int, str]:
    path, offset, total_bytes, shape = loc
    if shape[0] <= max(experts) or total_bytes != shape[0] * row_bytes * rows_per_expert:
        raise ValueError(f"unexpected expert tensor geometry for {path}: shape={shape} bytes={total_bytes}")
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as fh:
        for expert in experts:
            start = offset + expert * row_bytes * rows_per_expert
            fh.seek(start)
            payload = fh.read(row_bytes * rows_per_expert)
            if len(payload) != row_bytes * rows_per_expert:
                raise IOError(f"short expert row read at {path} offset {start}")
            digest.update(payload)
            bytes_read += len(payload)
    return bytes_read, digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc"))
    ap.add_argument("--layer-receipt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    layer = json.loads(args.layer_receipt.read_text())
    routes = layer["parity"]["route_ids_observed"]
    if len(routes) != 10 or len(set(routes)) != len(routes):
        raise ValueError("expected ten distinct verified route IDs")
    locations = load_locations(args.root, "model.safetensors.index.json")
    gate_name = "model.language_model.layers.44.mlp.experts.gate_up_proj"
    down_name = "model.language_model.layers.44.mlp.experts.down_proj"
    gate = locations[gate_name]
    down = locations[down_name]
    intermediate, hidden = 640, 2560
    full_gate_bytes = gate[2]
    full_down_bytes = down[2]
    selected_gate_bytes = len(routes) * (2 * intermediate * hidden * 2)
    selected_down_bytes = len(routes) * (hidden * intermediate * 2)
    started = time.perf_counter_ns()
    gate_read, gate_sha = read_expert_rows(gate, routes, hidden * 2, 2 * intermediate)
    down_read, down_sha = read_expert_rows(down, routes, intermediate * 2, hidden)
    elapsed = time.perf_counter_ns() - started
    full_bytes = full_gate_bytes + full_down_bytes
    selected_bytes = gate_read + down_read
    result = {
        "schema": "hawking.flash.expert_bank_routed_io_profile.v1",
        "status": "PROFILE_ONLY",
        "claim_boundary": "physical routed range-read profile for verified layer-44 route IDs; no compact-bank Metal execution or complete-token promotion",
        "layer": 44,
        "source": {
            "root": str(args.root),
            "layer_receipt": str(args.layer_receipt),
            "route_ids": routes,
            "gate_up_tensor": gate_name,
            "down_tensor": down_name,
        },
        "full_expert_bank_bytes": full_bytes,
        "selected_expert_bytes": selected_bytes,
        "bytes_reduction": full_bytes - selected_bytes,
        "selected_fraction": selected_bytes / full_bytes,
        "reduction_fraction": 1.0 - selected_bytes / full_bytes,
        "physical_range_read": {
            "bytes_read": gate_read + down_read,
            "elapsed_ns": elapsed,
            "gate_up_sha256": gate_sha,
            "down_sha256": down_sha,
        },
        "next_gate": "compact routed expert-bank Metal buffers plus route-ID-to-slot LUT, then exact organ parity",
        "promotion_allowed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
