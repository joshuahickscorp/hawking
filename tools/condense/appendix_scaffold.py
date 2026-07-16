#!/usr/bin/env python3.12
"""Non-executing registry for Hawking's post-ladder Appendix experiments."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import appendix_catalog
import appendix_contract
import appendix_corpus
import appendix_postrun
import spec_reentry_scaffold
import tq_runtime_probe
import tq_receipt_contract
import tq_runtime_matrix


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "hawking.appendix_plan.v1"
CORPUS_ROOT = ROOT / "reports" / "condense" / "doctor_v5_ultra" / "results"


def _id(family: str, knobs: dict) -> str:
    payload = json.dumps(
        {"family": family, "knobs": knobs}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"appendix-{hashlib.sha256(payload).hexdigest()[:12]}"


def _cell(
    family: str,
    currency: tuple[str, ...],
    knobs: dict,
    *,
    heavy: bool,
    depends_on: tuple[str, ...] = (),
) -> dict:
    return {
        "id": _id(family, knobs),
        "family": family,
        "currencies": list(currency),
        "knobs": knobs,
        "depends_on": list(depends_on),
        "requires_exclusive_heavy_lease": heavy,
        "mutates_active_corpus": False,
        "state": "deferred",
    }


def build_plan() -> dict:
    corpus = _cell(
        "freeze_corpus_index",
        ("state", "coordination"),
        {
            "source": "current_doctor_v5_ladder",
            "include_negative_evidence": True,
            "hash_large_artifacts_only_after_heavy_owner_exits": True,
        },
        heavy=False,
    )
    tq = _cell(
        "tq_compute_for_memory",
        ("arithmetic", "bytes"),
        {
            "runtime_paths": ["stored", "compact", "hashed", "computed", "repacked_lut"],
            "metrics": ["q12_parity", "wall_ns", "bytes", "occupancy", "energy"],
        },
        heavy=True,
        depends_on=(corpus["id"],),
    )
    tq_batch = _cell(
        "tq_batch_and_prefill_reuse",
        ("bytes", "coordination"),
        {
            "batch_sizes": [1, 2, 4, 8, 16, 32],
            "context_classes": ["short", "medium", "long"],
            "modes": ["decode", "chunked_prefill", "mixed_iteration"],
        },
        heavy=True,
        depends_on=(tq["id"],),
    )
    state = _cell(
        "state_and_token_avoidance",
        ("state", "bytes", "serial"),
        {
            "levers": ["exact_prefix", "kv_precision", "prompt_budget", "rag_budget", "output_budget"],
            "semantic_cache_status": "parked_control",
            "quality_and_output_length_held": True,
        },
        heavy=True,
        depends_on=(corpus["id"],),
    )
    spec = _cell(
        "exact_multi_token_commit",
        ("serial", "bytes", "coordination"),
        {
            "detail_schema": spec_reentry_scaffold.SCHEMA,
            "detail_cell_count": len(spec_reentry_scaffold.build_matrix("CORPUS")["cells"]),
            "requires_tq_native_verifier": True,
        },
        heavy=True,
        depends_on=(tq["id"],),
    )
    ledger = _cell(
        "capability_efficiency_ledger",
        ("arithmetic", "bytes", "communication", "serial", "coordination"),
        {
            "phases": ["load", "prefill", "draft", "verify", "decode", "kv", "sync", "io"],
            "metrics": [
                "quality",
                "accepted_tokens",
                "rejected_tokens",
                "p50_p95_p99",
                "bytes",
                "joules",
                "pressure",
                "swap",
                "thermal",
            ],
            "target_overhead_percent": 2.0,
        },
        heavy=True,
        depends_on=(corpus["id"],),
    )
    composition = _cell(
        "cross_appendix_composition",
        ("arithmetic", "bytes", "communication", "serial", "coordination"),
        {
            "promotion_metric": "quality_gated_accepted_tokens_per_second_and_joule",
            "factorial_slice": ["tq_runtime", "batch", "context", "kv", "proposer", "power"],
            "require_confidence_bounds": True,
        },
        heavy=True,
        depends_on=(tq_batch["id"], state["id"], spec["id"], ledger["id"]),
    )
    cells = [corpus, tq, tq_batch, state, spec, ledger, composition]
    return {
        "schema": SCHEMA,
        "name": "The Appendix",
        "capability_catalog_schema": appendix_catalog.SCHEMA,
        "capability_sector_count": len(appendix_catalog.build_catalog()["sectors"]),
        "receipt_schema": appendix_contract.SCHEMA,
        "corpus_index_schema": appendix_corpus.SCHEMA,
        "static_tq_probe_schema": tq_runtime_probe.SCHEMA,
        "tq_device_receipt_schema": tq_receipt_contract.SCHEMA,
        "tq_device_matrix_schema": tq_runtime_matrix.SCHEMA,
        "postrun_bridge_schema": appendix_postrun.SCHEMA,
        "execution_supported": False,
        "active_run_is_primary_corpus": True,
        "do_not_stop_or_modify_active_run": True,
        "currencies": ["arithmetic", "bytes", "state", "communication", "serial", "coordination"],
        "cells": cells,
    }


def corpus_snapshot(root: pathlib.Path = CORPUS_ROOT) -> dict:
    """Count paths only. Never open or hash an active artifact."""
    counts = {"request_receipts": 0, "result_receipts": 0, "tq_artifacts": 0}
    if root.is_dir():
        for dirpath, _dirnames, filenames in os.walk(root):
            del dirpath
            for name in filenames:
                if name == "request.json":
                    counts["request_receipts"] += 1
                elif name in {"result.json", "receipt.json"}:
                    counts["result_receipts"] += 1
                elif name.endswith(".tq"):
                    counts["tq_artifacts"] += 1
    return {"root": str(root), "exists": root.is_dir(), "counts": counts}


def _selftest() -> int:
    first = build_plan()
    second = build_plan()
    assert first == second
    assert first["execution_supported"] is False
    assert first["capability_sector_count"] == 25
    ids = [cell["id"] for cell in first["cells"]]
    assert len(ids) == len(set(ids))
    known = set(ids)
    for cell in first["cells"]:
        assert set(cell["depends_on"]) <= known
        assert cell["mutates_active_corpus"] is False
        assert cell["state"] == "deferred"
    print("appendix_scaffold.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        return _selftest()
    if argv == ["--plan"]:
        print(json.dumps(build_plan(), indent=2, sort_keys=True))
        return 0
    if argv == ["--status"]:
        owners = spec_reentry_scaffold.active_heavy_owners()
        print(
            json.dumps(
                {
                    "schema": "hawking.appendix_status.v1",
                    "corpus": corpus_snapshot(),
                    "active_heavy_owners": owners,
                    "safe_to_hash_or_run_heavy_cells": not owners,
                    "execution_supported": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    print("usage: appendix_scaffold.py --plan | --status | --selftest", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
