#!/usr/bin/env python3.12
"""Canonical Appendix profile, evidence, and compatibility runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import condense_profiles
import appendix_physical_counter_authority as authority
import appendix_physical_evidence_gate as evidence


SCHEMA = "hawking.appendix_condensed_runtime.v1"
PROFILE = {
    "sectors": 25,
    "runtime_paths": ["stored", "compact", "hashed", "computed"],
    "counter_domains": ["energy", "gpu_time", "physical_bytes", "occupancy", "bandwidth"],
    "physical_default_off": True,
    "structural_evidence_points": 0,
    "cpu_error_policy": {
        "max_abs_error": 1.0e-4,
        "max_rel_error": 1.0e-4,
    },
    "canonical_modules": [
        "appendix_runtime.py",
        "appendix_physical_evidence_gate.py",
        "appendix_physical_counter_authority.py",
        "physical_counter_attestation.py",
    ],
}


def validate(document: Any, *, verify_files: bool = True) -> list[str]:
    return evidence.validate_gate(document, verify_counter_files=verify_files)


def replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("cell_id"), str):
            raise ValueError(f"rows[{index}] must bind cell_id")
        latest[row["cell_id"]] = dict(row)
    statuses = {
        status: sum(row.get("status") == status for row in latest.values())
        for status in ("pending", "running", "complete", "negative", "unsupported")
    }
    return {
        "schema": "hawking.appendix_condensed_replay.v1",
        "cells": [latest[key] for key in sorted(latest)],
        "status_counts": statuses,
        "terminal": sum(statuses[key] for key in ("complete", "negative", "unsupported")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profile")
    sub.add_parser("requirements")
    check = sub.add_parser("validate")
    check.add_argument("packet", type=Path)
    check.add_argument("--no-file-verification", action="store_true")
    legacy = sub.add_parser("legacy")
    legacy.add_argument("module")
    legacy.add_argument("arguments", nargs=argparse.REMAINDER)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "profile":
        print(json.dumps({"schema": SCHEMA, **PROFILE}, indent=2, sort_keys=True))
        return 0
    if args.command == "requirements":
        print(json.dumps(evidence.requirements(), indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        document = json.loads(args.packet.read_text(encoding="utf-8"))
        errors = validate(document, verify_files=not args.no_file_verification)
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    if args.command == "legacy":
        return condense_profiles.invoke(args.module, args.arguments)
    assert validate({}) and authority.SIGNER_IDENTITY == "hawking-appendix-release"
    assert replay([{"cell_id": "a", "status": "complete"}])["terminal"] == 1
    print("appendix_runtime.py selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
