#!/usr/bin/env python3.12
"""Cheap analytic TQ runtime byte/ALU probe; reads no model or active artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
from typing import Any


SCHEMA = "hawking.tq_runtime_static_probe.v1"
TAIL_LEFT_LEN = (0, 1, 2, 3, 6, 12, 25, 50, 99, 199, 397)
MODEL_CONFIGS = (
    {"model": "qwen_0_5b", "hidden": 896, "intermediate": 4864, "layers": 24, "heads": 14, "kv_heads": 2, "vocab": 151936},
    {"model": "qwen_1_5b", "hidden": 1536, "intermediate": 8960, "layers": 28, "heads": 12, "kv_heads": 2, "vocab": 151936},
    {"model": "qwen_3b", "hidden": 2048, "intermediate": 11008, "layers": 36, "heads": 16, "kv_heads": 2, "vocab": 151936},
    {"model": "qwen_7b", "hidden": 3584, "intermediate": 18944, "layers": 28, "heads": 28, "kv_heads": 4, "vocab": 152064},
    {"model": "qwen_14b", "hidden": 5120, "intermediate": 13824, "layers": 48, "heads": 40, "kv_heads": 8, "vocab": 152064},
    {"model": "qwen_32b", "hidden": 5120, "intermediate": 27648, "layers": 64, "heads": 40, "kv_heads": 8, "vocab": 152064},
    {"model": "qwen_72b", "hidden": 8192, "intermediate": 29568, "layers": 80, "heads": 64, "kv_heads": 8, "vocab": 152064},
)


def _default_shapes() -> tuple[dict, ...]:
    shapes: list[dict] = []
    for cfg in MODEL_CONFIGS:
        model = cfg["model"]
        hidden = cfg["hidden"]
        intermediate = cfg["intermediate"]
        layers = cfg["layers"]
        head_dim = hidden // cfg["heads"]
        kv_width = head_dim * cfg["kv_heads"]
        shapes.extend((
            {"model": model, "label": f"{model}_ffn_up_gate", "rows": intermediate, "cols": hidden, "multiplicity": 2 * layers},
            {"model": model, "label": f"{model}_ffn_down", "rows": hidden, "cols": intermediate, "multiplicity": layers},
            {"model": model, "label": f"{model}_attn_q_o", "rows": hidden, "cols": hidden, "multiplicity": 2 * layers},
            {"model": model, "label": f"{model}_attn_k_v", "rows": kv_width, "cols": hidden, "multiplicity": 2 * layers},
            {"model": model, "label": f"{model}_lm_head", "rows": cfg["vocab"], "cols": hidden, "multiplicity": 1},
        ))
    shapes.extend((
        {"model": "gpt_oss_120b_proxy", "label": "gpt_oss_120b_dense_projection_proxy", "rows": 2880, "cols": 2880, "multiplicity": 36},
        {"model": "synthetic", "label": "square_4k", "rows": 4096, "cols": 4096, "multiplicity": 1},
    ))
    return tuple(shapes)


DEFAULT_SHAPES = _default_shapes()
MODES = {
    "stored": {"metadata": 84, "codebook_bytes_per_entry": 4, "tail": False, "ops": 2, "implemented": True},
    "compact": {"metadata": 40, "codebook_bytes_per_entry": 4, "tail": False, "ops": 3, "implemented": True},
    "hashed": {"metadata": 84, "codebook_bytes_per_entry": 2, "tail": False, "ops": 8, "implemented": True},
    "computed": {"metadata": 84, "codebook_bytes_per_entry": 0, "tail": True, "ops": 24, "implemented": True},
    "compact_hashed": {"metadata": 40, "codebook_bytes_per_entry": 2, "tail": False, "ops": 9, "implemented": False},
    "compact_computed": {"metadata": 40, "codebook_bytes_per_entry": 0, "tail": True, "ops": 25, "implemented": False},
    "repacked_lut": {"metadata": 40, "codebook_bytes_per_entry": 4, "tail": False, "ops": 2, "implemented": False},
}


def _sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def probe_cell(
    rows: int,
    cols: int,
    k_bits: int,
    l_bits: int,
    mode: str,
    label: str,
    *,
    model: str = "synthetic",
    multiplicity: int = 1,
) -> dict:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if k_bits not in range(1, 5) or l_bits not in range(4, 15) or l_bits < k_bits:
        raise ValueError("requires k=1..4 and max(k,4)<=L<=14")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    if multiplicity <= 0:
        raise ValueError("multiplicity must be positive")
    cfg = MODES[mode]
    weights = rows * cols
    blocks_per_row = math.ceil(cols / 256)
    blocks = rows * blocks_per_row
    threadgroups = math.ceil(blocks / 256)
    payload_bytes = math.ceil(weights * k_bits / 8)
    metadata_bytes = blocks * cfg["metadata"]
    if cfg["tail"]:
        codebook_per_group = TAIL_LEFT_LEN[l_bits - 4] * 4
    else:
        codebook_per_group = (1 << l_bits) * cfg["codebook_bytes_per_entry"]
    codebook_staging_bytes = threadgroups * codebook_per_group
    partial_roundtrip_bytes = blocks * 8
    activation_logical_bytes = weights * 4
    output_bytes = rows * 4
    compressed_path = payload_bytes + metadata_bytes + codebook_staging_bytes + partial_roundtrip_bytes
    total_logical = compressed_path + activation_logical_bytes + output_bytes
    return {
        "id": _sha({"rows": rows, "cols": cols, "k": k_bits, "l": l_bits, "mode": mode, "label": label, "multiplicity": multiplicity})[:16],
        "model": model,
        "label": label,
        "multiplicity": multiplicity,
        "shape": {"rows": rows, "cols": cols},
        "k_bits": k_bits,
        "l_bits": l_bits,
        "mode": mode,
        "implemented": cfg["implemented"],
        "weights": weights,
        "blocks": blocks,
        "blocks_per_row": blocks_per_row,
        "ragged_tail_weights_per_row": cols % 256,
        "current_gpu_fused_gemv_eligible": cols % 256 == 0,
        "threadgroups": threadgroups,
        "logical_bytes": {
            "payload": payload_bytes,
            "metadata": metadata_bytes,
            "codebook_staging": codebook_staging_bytes,
            "partial_roundtrip": partial_roundtrip_bytes,
            "activations": activation_logical_bytes,
            "output": output_bytes,
            "compressed_path_total": compressed_path,
            "all_listed_total": total_logical,
        },
        "bpw": {
            "payload": payload_bytes * 8 / weights,
            "payload_plus_metadata": (payload_bytes + metadata_bytes) * 8 / weights,
            "compressed_path": compressed_path * 8 / weights,
        },
        "analytic_integer_ops_per_weight": cfg["ops"],
        "multiplicity_weighted": {
            "weights": weights * multiplicity,
            "compressed_path_bytes": compressed_path * multiplicity,
            "all_listed_bytes": total_logical * multiplicity,
        },
        "measurement_status": "analytic_not_physical",
    }


def _dominates(left: dict, right: dict) -> bool:
    lb = left["logical_bytes"]["compressed_path_total"]
    rb = right["logical_bytes"]["compressed_path_total"]
    lo = left["analytic_integer_ops_per_weight"]
    ro = right["analytic_integer_ops_per_weight"]
    return lb <= rb and lo <= ro and (lb < rb or lo < ro)


def build_probe() -> dict:
    cells = [
        probe_cell(
            shape["rows"],
            shape["cols"],
            k,
            k + 6,
            mode,
            shape["label"],
            model=shape["model"],
            multiplicity=shape["multiplicity"],
        )
        for shape in DEFAULT_SHAPES
        for k in range(1, 5)
        for mode in MODES
    ]
    groups: list[dict] = []
    for shape in DEFAULT_SHAPES:
        for k in range(1, 5):
            members = [cell for cell in cells if cell["label"] == shape["label"] and cell["k_bits"] == k]
            frontier = [cell["mode"] for cell in members if not any(_dominates(other, cell) for other in members if other is not cell)]
            groups.append({"label": shape["label"], "k_bits": k, "pareto_modes": sorted(frontier)})
    model_rollups = []
    for model in sorted({cell["model"] for cell in cells if cell["model"] != "synthetic"}):
        for k in range(1, 5):
            for mode in MODES:
                members = [
                    cell for cell in cells
                    if cell["model"] == model and cell["k_bits"] == k and cell["mode"] == mode
                ]
                model_rollups.append({
                    "model": model,
                    "k_bits": k,
                    "mode": mode,
                    "implemented": MODES[mode]["implemented"],
                    "projection_family_count": len(members),
                    "multiplicity_weighted_weights": sum(cell["multiplicity_weighted"]["weights"] for cell in members),
                    "multiplicity_weighted_compressed_path_bytes": sum(cell["multiplicity_weighted"]["compressed_path_bytes"] for cell in members),
                    "all_projection_families_gpu_eligible": all(cell["current_gpu_fused_gemv_eligible"] for cell in members),
                    "ineligible_projection_families": [cell["label"] for cell in members if not cell["current_gpu_fused_gemv_eligible"]],
                })
    return {
        "schema": SCHEMA,
        "execution_kind": "static_analytic",
        "reads_model_artifacts": False,
        "safe_during_active_run": True,
        "assumptions": {
            "scalar_row_major_block_weights": 256,
            "threads_per_group": 256,
            "expanded_metadata_bytes_per_block": 84,
            "compact_metadata_bytes_per_block": 40,
            "partial_roundtrip_bytes_per_block": 8,
            "activation_bytes_per_weight_is_logical_not_cache_traffic": 4,
            "integer_ops_are_ordering_proxies_not_instruction_counts": True,
            "physical_bandwidth_occupancy_latency_and_energy_unmeasured": True,
        },
        "cells": cells,
        "pareto_groups": groups,
        "model_rollups": model_rollups,
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
    baseline = probe_cell(1, 256, 3, 9, "stored", "test")
    compact = probe_cell(1, 256, 3, 9, "compact", "test")
    assert baseline["bpw"]["payload_plus_metadata"] == 5.625
    assert compact["bpw"]["payload_plus_metadata"] == 4.25
    assert baseline["logical_bytes"]["metadata"] - compact["logical_bytes"]["metadata"] == 44
    assert probe_cell(2, 257, 3, 9, "stored", "ragged")["current_gpu_fused_gemv_eligible"] is False
    assert build_probe() == build_probe()
    assert len(build_probe()["cells"]) == len(DEFAULT_SHAPES) * 4 * len(MODES)
    print("tq_runtime_probe.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--write", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    probe = build_probe()
    if args.write is not None:
        _atomic_json(args.write, probe)
        return 0
    if args.probe:
        print(json.dumps(probe, indent=2, sort_keys=True))
        return 0
    parser.error("one of --probe, --write, or --selftest is required")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
