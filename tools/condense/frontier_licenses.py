#!/usr/bin/env python3.12
"""frontier_licenses.py - strict license/gating approval checks for frontier procurement."""
from __future__ import annotations

import pathlib
from typing import Any

COND_DIR = pathlib.Path("reports/condense")
LICENSE_PATH = COND_DIR / "frontier_license_acceptance.json"

ALLOWED_USE = {"research", "internal-research", "personal-research", "evaluation"}
REDISTRIBUTION = {"none", "artifact-only", "per-license"}
SOURCE_POLICY = {"local-only-delete-after-bake", "local-only-retain-per-license", "per-license"}


def _problem(record: dict[str, Any], key: str, msg: str) -> str | None:
    return None if record.get(key) else msg


def license_status(record: dict[str, Any] | None, label: str) -> dict[str, Any]:
    problems = []
    if not record:
        return {
            "label": label,
            "status": "unreviewed",
            "ok": False,
            "problems": ["license/gating approval is missing"],
        }
    status = record.get("status", "unreviewed")
    if status == "rejected":
        return {
            "label": label,
            "status": status,
            "ok": False,
            "problems": ["license review rejected"],
            "record": record,
        }
    if status != "accepted":
        return {
            "label": label,
            "status": status,
            "ok": False,
            "problems": [f"status must be accepted before procurement, got {status!r}"],
            "record": record,
        }
    for maybe in (
        _problem(record, "by", "accepted license record needs --by"),
        _problem(record, "license", "accepted license record needs --license"),
        _problem(record, "terms_url", "accepted license record needs --terms-url"),
        _problem(record, "allowed_use", "accepted license record needs --allowed-use"),
        _problem(record, "redistribution", "accepted license record needs --redistribution"),
        _problem(record, "source_policy", "accepted license record needs --source-policy"),
        _problem(record, "note", "accepted license record needs --note"),
    ):
        if maybe:
            problems.append(maybe)
    if record.get("allowed_use") and record["allowed_use"] not in ALLOWED_USE:
        problems.append(f"allowed_use must be one of {sorted(ALLOWED_USE)}")
    if record.get("redistribution") and record["redistribution"] not in REDISTRIBUTION:
        problems.append(f"redistribution must be one of {sorted(REDISTRIBUTION)}")
    if record.get("source_policy") and record["source_policy"] not in SOURCE_POLICY:
        problems.append(f"source_policy must be one of {sorted(SOURCE_POLICY)}")
    return {
        "label": label,
        "status": status,
        "ok": not problems,
        "problems": problems,
        "record": record,
    }


def license_rollup(ledger: dict[str, dict], labels: list[str]) -> dict[str, Any]:
    rows = [license_status(ledger.get(label), label) for label in labels]
    blocked = [row["label"] for row in rows if not row["ok"]]
    return {
        "schema": "hawking.frontier_license_rollup.v1",
        "model_count": len(rows),
        "passed_count": len(rows) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "rows": rows,
        "ok": not blocked,
    }


def license_plan(labels: list[str]) -> dict[str, Any]:
    return {
        "schema": "hawking.frontier_license_plan.v1",
        "allowed_use": sorted(ALLOWED_USE),
        "redistribution": sorted(REDISTRIBUTION),
        "source_policy": sorted(SOURCE_POLICY),
        "labels": [
            {
                "label": label,
                "command": (
                    "hawking studio record-license "
                    f"{label} --status accepted --by <name> --license <id> "
                    "--terms-url <url> --allowed-use research --redistribution none "
                    "--source-policy local-only-delete-after-bake --note <decision>"
                ),
            }
            for label in labels
        ],
    }
