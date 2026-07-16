#!/usr/bin/env python3.12
"""Receipt wrapping and non-comparative evidence rollup for The Appendix."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import appendix_contract as contract
import appendix_scaffold
import tq_runtime_probe


ROOT = pathlib.Path(__file__).resolve().parents[2]
ROLLUP_SCHEMA = "hawking.appendix_evidence_rollup.v1"


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _tq_cell_id() -> str:
    for cell in appendix_scaffold.build_plan()["cells"]:
        if cell["family"] == "tq_compute_for_memory":
            return cell["id"]
    raise AssertionError("master Appendix plan has no tq_compute_for_memory cell")


def static_probe_receipt(probe: dict, *, source_commit: str | None = None) -> dict:
    if probe.get("schema") != tq_runtime_probe.SCHEMA:
        raise ValueError(f"probe schema must be {tq_runtime_probe.SCHEMA}")
    cells = probe.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("probe must contain non-empty cells")
    samples: list[dict] = []
    for index, cell in enumerate(cells):
        logical = cell["logical_bytes"]
        samples.append({
            "index": index,
            "phase": "static",
            "wall_ns": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "energy_j": None,
            "bytes": {
                "weights": int(logical["payload"]),
                "metadata": int(logical["metadata"]),
                "codebook": int(logical["codebook_staging"]),
                "activations": int(logical["activations"]),
                "scratch": int(logical["partial_roundtrip"]),
                "state": 0,
                "communication": 0,
                "io": 0,
            },
            "probe_cell_id": cell["id"],
            "mode": cell["mode"],
            "model": cell["model"],
            "multiplicity": cell["multiplicity"],
            "implemented": cell["implemented"],
            "current_gpu_fused_gemv_eligible": cell["current_gpu_fused_gemv_eligible"],
            "shape": cell["shape"],
            "k_bits": cell["k_bits"],
            "l_bits": cell["l_bits"],
            "bpw": cell["bpw"],
            "analytic_integer_ops_per_weight": cell["analytic_integer_ops_per_weight"],
        })
    receipt = {
        "schema": contract.SCHEMA,
        "experiment_schema": tq_runtime_probe.SCHEMA,
        "cell_id": _tq_cell_id(),
        "status": "complete",
        "bindings": {
            "source_commit": source_commit or _source_commit(),
            "target_sha256": None,
            "target_not_applicable_reason": "static geometry probe reads no model artifact",
            "prompt_set_sha256": None,
            "prompt_not_applicable_reason": "static geometry probe consumes no prompts",
            "parent_receipt_sha256": [],
        },
        "exactness": {"required": False, "passed": None, "mismatches": None},
        "samples": samples,
        "rollup": contract.rollup_samples(samples),
        "resources": {
            "observed": False,
            "memory_pressure": None,
            "swap_delta_mb": None,
            "thermal_state": None,
            "not_applicable_reason": "analytic calculation performs no device workload",
        },
        "failure_reasons": [],
        "analytic_scope": {
            "probe_sha256": contract.canonical_sha256(probe),
            "cell_count": len(cells),
            "implemented_cells": sum(bool(cell["implemented"]) for cell in cells),
            "future_cells": sum(not bool(cell["implemented"]) for cell in cells),
            "gpu_eligible_cells": sum(bool(cell["current_gpu_fused_gemv_eligible"]) for cell in cells),
            "gpu_ineligible_cells": sum(not bool(cell["current_gpu_fused_gemv_eligible"]) for cell in cells),
            "physical_speed_claim": False,
        },
    }
    stamped = contract.stamp_receipt(receipt)
    errors = contract.validate_receipt(stamped, known_cell_ids={_tq_cell_id()})
    if errors:
        raise ValueError("invalid generated receipt: " + "; ".join(errors))
    return stamped


def rollup_receipts(receipts: list[dict]) -> dict:
    rows = []
    byte_totals = {field: 0 for field in contract.BYTE_FIELDS}
    accepted = 0
    rejected = 0
    for receipt in receipts:
        errors = contract.validate_receipt(receipt)
        rows.append({
            "cell_id": receipt.get("cell_id"),
            "experiment_schema": receipt.get("experiment_schema"),
            "status": receipt.get("status"),
            "receipt_sha256": receipt.get("receipt_sha256"),
            "valid": not errors,
            "errors": errors,
        })
        if not errors and receipt.get("status") == "complete":
            summary = receipt["rollup"]
            accepted += summary["accepted_tokens"]
            rejected += summary["rejected_tokens"]
            for field in contract.BYTE_FIELDS:
                byte_totals[field] += summary["bytes"][field]
    return {
        "schema": ROLLUP_SCHEMA,
        "receipt_count": len(receipts),
        "valid_count": sum(row["valid"] for row in rows),
        "invalid_count": sum(not row["valid"] for row in rows),
        "status_counts": {
            state: sum(row["status"] == state for row in rows)
            for state in ("deferred", "complete", "failed")
        },
        "raw_totals_not_cross_cell_performance_claims": {
            "accepted_tokens": accepted,
            "rejected_tokens": rejected,
            "bytes": byte_totals,
        },
        "receipts": rows,
    }


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    receipt = static_probe_receipt(tq_runtime_probe.build_probe(), source_commit="0123456789abcdef")
    assert contract.validate_receipt(receipt) == []
    assert receipt["analytic_scope"]["cell_count"] == len(tq_runtime_probe.DEFAULT_SHAPES) * 4 * len(tq_runtime_probe.MODES)
    rollup = rollup_receipts([receipt])
    assert rollup["valid_count"] == 1
    assert rollup["invalid_count"] == 0
    assert rollup["status_counts"]["complete"] == 1
    print("appendix_ledger.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrap-static-probe", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--rollup", nargs="*", type=pathlib.Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.wrap_static_probe is not None:
        if args.output is None:
            parser.error("--wrap-static-probe requires --output")
        _atomic_json(args.output, static_probe_receipt(_load(args.wrap_static_probe)))
        return 0
    if args.rollup is not None:
        value = rollup_receipts([_load(path) for path in args.rollup])
        if args.output is None:
            print(json.dumps(value, indent=2, sort_keys=True))
        else:
            _atomic_json(args.output, value)
        return 0 if value["invalid_count"] == 0 else 1
    parser.error("choose --wrap-static-probe, --rollup, or --selftest")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
