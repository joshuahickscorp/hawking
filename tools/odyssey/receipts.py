#!/usr/bin/env python3.12
"""Content-addressed training receipts.

Every stage that runs through the apparatus emits one receipt whose id is the
sha256 of its canonical body. Metadata like wall_clock may be attached after
hashing or stored outside the hashed body so identity stays content-addressed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

SCHEMA = "hawking.odyssey.training_receipt.v1"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_id(body: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(body)).hexdigest()


def make_receipt(
    *,
    stage: str,
    status: str,
    checkpoint_id: str | None,
    objective: str | None,
    steps_completed: int,
    state_sha256: str | None,
    fixture: bool,
    details: dict[str, Any] | None = None,
    parent_receipt_id: str | None = None,
) -> dict[str, Any]:
    """Build a content-addressed receipt. wall_clock is outside the hashed body."""
    body = {
        "schema": SCHEMA,
        "stage": stage,
        "status": status,
        "checkpoint_id": checkpoint_id,
        "objective": objective,
        "steps_completed": int(steps_completed),
        "state_sha256": state_sha256,
        "fixture": bool(fixture),
        "details": details or {},
        "parent_receipt_id": parent_receipt_id,
    }
    if fixture:
        body["fixture_label"] = (
            "FIXTURE: apparatus receipt over a toy model; never trained anything real"
        )
    rid = content_id(body)
    receipt = dict(body)
    receipt["receipt_id"] = rid
    receipt["wall_clock"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return receipt


def write_receipt(receipt: dict[str, Any], directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rid = receipt["receipt_id"]
    path = directory / f"{rid[:16]}.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (directory / "LATEST").write_text(rid + "\n")
    return path


def verify_receipt(receipt: dict[str, Any]) -> bool:
    """Recompute content id from body fields; wall_clock is not part of identity."""
    body = {
        k: receipt[k]
        for k in (
            "schema",
            "stage",
            "status",
            "checkpoint_id",
            "objective",
            "steps_completed",
            "state_sha256",
            "fixture",
            "details",
            "parent_receipt_id",
        )
        if k in receipt
    }
    if receipt.get("fixture") and "fixture_label" in receipt:
        body["fixture_label"] = receipt["fixture_label"]
    return content_id(body) == receipt.get("receipt_id")
