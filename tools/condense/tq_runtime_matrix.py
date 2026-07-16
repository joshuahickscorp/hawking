#!/usr/bin/env python3.12
"""Deterministic post-run TQ device matrix derived from the static census."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import appendix_contract
import tq_receipt_contract
import tq_runtime_probe


SCHEMA = "hawking.tq_runtime_device_matrix.v1"


def _id(cell: dict) -> str:
    identity = {
        "label": cell["label"],
        "rows": cell["shape"]["rows"],
        "cols": cell["shape"]["cols"],
        "k_bits": cell["k_bits"],
        "l_bits": cell["l_bits"],
        "mode": cell["mode"],
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "tqdev-" + hashlib.sha256(raw).hexdigest()[:16]


def build_matrix(probe: dict | None = None) -> dict:
    probe = tq_runtime_probe.build_probe() if probe is None else probe
    if probe.get("schema") != tq_runtime_probe.SCHEMA:
        raise ValueError(f"probe schema must be {tq_runtime_probe.SCHEMA}")
    source_cells = probe["cells"]
    id_by_key = {
        (cell["label"], cell["k_bits"], cell["mode"]): _id(cell)
        for cell in source_cells
    }
    cells = []
    for source in source_cells:
        mode = source["mode"]
        implemented = source["implemented"]
        eligible = source["current_gpu_fused_gemv_eligible"]
        if not implemented:
            state = "design_deferred"
            blocker = "runtime recipe has no admitted Metal kernel"
        elif not eligible:
            state = "blocked_geometry"
            blocker = "current fused GEMV requires cols % 256 == 0"
        else:
            state = "deferred"
            blocker = None
        dependencies: list[str] = []
        if mode != "stored":
            dependencies.append(id_by_key[(source["label"], source["k_bits"], "stored")])
        cells.append({
            "id": _id(source),
            "probe_cell_id": source["id"],
            "model": source["model"],
            "tensor_family": source["label"],
            "multiplicity": source["multiplicity"],
            "shape": source["shape"],
            "k_bits": source["k_bits"],
            "l_bits": source["l_bits"],
            "runtime_path": mode if implemented else None,
            "research_recipe": mode,
            "implemented": implemented,
            "gpu_eligible": eligible,
            "state": state,
            "blocker": blocker,
            "depends_on": dependencies,
            "receipt_schema": tq_receipt_contract.SCHEMA if implemented else None,
            "requires_exclusive_heavy_lease": True,
            "mutates_active_corpus": False,
            "gates": [
                "metal_compile",
                "entry_size_identity",
                "q12_parity",
                "fused_gemv_parity",
                "warm_latency_distribution",
                "logical_and_physical_bytes",
                "occupancy_bandwidth_energy_resources",
            ],
        })
    known = {cell["id"] for cell in cells}
    if any(not set(cell["depends_on"]) <= known for cell in cells):
        raise AssertionError("matrix dependency closure failed")
    counts = {
        state: sum(cell["state"] == state for cell in cells)
        for state in ("deferred", "blocked_geometry", "design_deferred")
    }
    return {
        "schema": SCHEMA,
        "source_probe_schema": tq_runtime_probe.SCHEMA,
        "source_probe_sha256": appendix_contract.canonical_sha256(probe),
        "receipt_schema": tq_receipt_contract.SCHEMA,
        "execution_supported": False,
        "start_only_after_heavy_owner_exits": True,
        "stored_baseline_required_per_shape_and_k": True,
        "counts": {"total": len(cells), **counts},
        "cells": cells,
    }


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _selftest() -> int:
    first = build_matrix()
    assert first == build_matrix()
    assert first["counts"]["total"] == len(tq_runtime_probe.DEFAULT_SHAPES) * 4 * len(tq_runtime_probe.MODES)
    assert sum(first["counts"][state] for state in ("deferred", "blocked_geometry", "design_deferred")) == first["counts"]["total"]
    assert first["execution_supported"] is False
    assert len({cell["id"] for cell in first["cells"]}) == first["counts"]["total"]
    for cell in first["cells"]:
        assert cell["mutates_active_corpus"] is False
        if cell["runtime_path"] not in {None, "stored"}:
            assert len(cell["depends_on"]) == 1
    print("tq_runtime_matrix.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--write", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    matrix = build_matrix()
    if args.write is not None:
        _atomic_json(args.write, matrix)
        return 0
    if args.matrix:
        print(json.dumps(matrix, indent=2, sort_keys=True))
        return 0
    parser.error("choose --matrix, --write, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
