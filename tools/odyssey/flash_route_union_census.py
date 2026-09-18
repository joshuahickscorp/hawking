#!/usr/bin/env python3
"""Derive an exact bounded Flash route-union control from a dense session.

This is deliberately a *teacher-bound* control: it uses the native top-k IDs
recorded by an exact dense source session, validates every layer/token row, and
then bills the union's expert-bank bytes.  It neither predicts future routes
nor claims a reusable resident cache.  Its only purpose is to make the first
route-safe compact execution candidate falsifiable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
LAYERS = 48
EXPERTS = 512
TOP_K = 10


def _load(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"receipt root must be an object: {path}")
    return doc


def _route_rows(session: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in session.get("segments", []):
        if not isinstance(segment, dict):
            raise ValueError("session segment must be an object")
        if isinstance(segment.get("steps"), list):
            rows.extend(segment["steps"])
            continue
        receipt = segment.get("receipt")
        if not isinstance(receipt, str):
            raise ValueError("nonlinear segment omitted its route receipt")
        layer = segment.get("layer")
        if not isinstance(layer, int) or not 0 <= layer < LAYERS:
            raise ValueError("nonlinear segment omitted its Flash layer coordinate")
        path = Path(receipt)
        if not path.is_absolute():
            path = root / path
        nested = _load(path)
        nested_rows = nested.get("steps")
        if not isinstance(nested_rows, list):
            raise ValueError(f"attention route receipt omitted steps: {path}")
        # Attention-organ receipts carry token slots but inherit their layer
        # coordinate from the enclosing session segment.  Stamp that envelope
        # coordinate before validating the single session-wide route grid.
        for row in nested_rows:
            if not isinstance(row, dict):
                raise ValueError(f"attention route row must be an object: {path}")
            stamped = dict(row)
            recorded_layer = stamped.get("layer")
            if recorded_layer is not None and recorded_layer != layer:
                raise ValueError(f"attention route row disagrees with segment layer: {path}")
            stamped["layer"] = layer
            rows.append(stamped)
    return rows


def derive(session: dict[str, Any], root: Path, expert_family_bytes: int) -> dict[str, Any]:
    if session.get("status") != "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE":
        raise ValueError("route-union teacher must be a passed repeated accepted session")
    if session.get("model") != REPO_ID or session.get("pinned_revision") != PINNED_REVISION:
        raise ValueError("route-union teacher source identity drifted")
    tokens = session.get("token_ids")
    if not isinstance(tokens, list) or len(tokens) < 2 or not all(isinstance(v, int) for v in tokens):
        raise ValueError("route-union teacher omitted integer token IDs")
    raw_rows = _route_rows(session, root)
    expected = {(layer, step) for layer in range(LAYERS) for step in range(len(tokens))}
    unions: dict[int, set[int]] = {layer: set() for layer in range(LAYERS)}
    seen: set[tuple[int, int]] = set()
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("route row must be an object")
        layer, step, routes = row.get("layer"), row.get("step"), row.get("route_ids")
        if not isinstance(layer, int) or not isinstance(step, int) or (layer, step) not in expected:
            raise ValueError("route row has an unexpected layer/token coordinate")
        if (layer, step) in seen:
            raise ValueError("route row duplicated a layer/token coordinate")
        if not isinstance(routes, list) or len(routes) != TOP_K:
            raise ValueError("route row omitted the exact top-k selection")
        if not all(isinstance(expert, int) and 0 <= expert < EXPERTS for expert in routes):
            raise ValueError("route row contains an invalid expert ID")
        if len(set(routes)) != TOP_K:
            raise ValueError("route row contains duplicate top-k expert IDs")
        seen.add((layer, step))
        unions[layer].update(routes)
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"route coverage incomplete; first missing={missing[:1]}")
    union_rows = [
        {
            "layer": layer,
            "experts": sorted(unions[layer]),
            "expert_count": len(unions[layer]),
            "route_rows": len(tokens),
        }
        for layer in range(LAYERS)
    ]
    selected_rows = sum(row["expert_count"] for row in union_rows)
    total_rows = LAYERS * EXPERTS
    compact_expert_bytes = expert_family_bytes * selected_rows // total_rows
    doc = {
        "schema": "hawking.flash.route_union_census.v1",
        "status": "MEASURED_DENSE_TEACHER_ROUTE_UNION",
        "model": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "teacher_contract": {
            "status": session.get("status"),
            "token_ids": tokens,
            "accepted_generation_tokens": session.get("accepted_generation_tokens"),
            "source_reset_or_reprefill": session.get("execution", {}).get("source_reset_or_reprefill"),
            "process_boundary": session.get("execution", {}).get("process_boundary"),
        },
        "coverage": {
            "layers": LAYERS,
            "token_slots": len(tokens),
            "top_k": TOP_K,
            "expected_route_rows": len(expected),
            "observed_route_rows": len(seen),
            "complete": True,
        },
        "expert_union": {
            "full_rows": total_rows,
            "selected_rows": selected_rows,
            "selected_fraction": selected_rows / total_rows,
            "per_layer": union_rows,
        },
        "expert_bank_byte_control": {
            "dense_family_bytes": expert_family_bytes,
            "teacher_union_bf16_bytes": compact_expert_bytes,
            "teacher_union_fraction": compact_expert_bytes / expert_family_bytes,
            "reduction_factor": expert_family_bytes / compact_expert_bytes if compact_expert_bytes else None,
        },
        "promotion_allowed": False,
        "claim_boundary": "This bills a route union observed in one exact dense teacher session. It is not a future-route predictor, general cache policy, capability result, complete EBPW result, or TPS claim. A compact candidate must independently reproduce every observed route and accepted terminal token.",
        "next": "Run the same bounded token sequence through compact unions supplied by this receipt, reject any missing/changed route, and compare accepted terminal tokens plus bytes/dispatches against the dense teacher.",
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--session",
        type=Path,
        default=ROOT / "receipts/headless/FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json",
    )
    ap.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "receipts/headless/FLASH_COMPLETE_V0.BYTE_LEDGER.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "receipts/headless/FLASH_ROUTE_UNION_CENSUS.json",
    )
    args = ap.parse_args(argv)
    session_path = args.session.resolve()
    ledger = _load(args.ledger.resolve())
    family = ledger.get("routed_expert_sensitivity", {}).get("full_family_bytes")
    if not isinstance(family, int) or family <= 0:
        raise ValueError("byte ledger omitted routed expert family bytes")
    doc = derive(_load(session_path), ROOT, family)
    doc["teacher_receipt"] = str(session_path)
    doc["teacher_receipt_sha256"] = hashlib.sha256(session_path.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({
        "status": doc["status"],
        "selected_rows": doc["expert_union"]["selected_rows"],
        "full_rows": doc["expert_union"]["full_rows"],
        "teacher_union_bf16_bytes": doc["expert_bank_byte_control"]["teacher_union_bf16_bytes"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
