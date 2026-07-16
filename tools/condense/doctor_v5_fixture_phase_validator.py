#!/usr/bin/env python3.12
"""Strict cheap validator used only by inert-launcher contract fixtures.

Real Doctor adapters must bind their own phase-specific validator executable to
the same receipt schema.  This fixture proves the scheduler's parity/zero-skip
authority path without reading models, using a GPU, or touching corpus results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


OUTPUT_SCHEMA = "hawking.doctor_v5_fixture_phase_output.v1"
RECEIPT_SCHEMA = "hawking.doctor_v5_phase_semantic_validator_receipt.v1"
REQUEST_SCHEMA = "hawking.doctor_v5_inert_phase_launch_request.v1"
VERSION = "2026-07-14.1"
ROOT = Path(__file__).resolve().parents[2]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def reference(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(ROOT.resolve(strict=True)))
    except ValueError:
        display = str(resolved)
    return {"path": display,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_bytes())
        output = json.loads(args.output.read_bytes())
        claim_handshake = json.loads(Path(
            request["paths"]["target_claim_handshake"]
        ).read_bytes())
        claim_ack = json.loads(Path(
            request["paths"]["target_claim_ack"]
        ).read_bytes())
        resource_guard = json.loads(Path(
            request["paths"]["target_resource_guard"]
        ).read_bytes())
        claim_sha = hash_value(request.get("resource_claim"))
        if not isinstance(request, dict) or not isinstance(output, dict) \
                or request.get("schema") != REQUEST_SCHEMA \
                or request.get("request_sha256") != hash_value({
                    key: value for key, value in request.items()
                    if key != "request_sha256"
                }) \
                or output.get("schema") != OUTPUT_SCHEMA \
                or output.get("version") != VERSION \
                or output.get("phase") != request.get("phase") \
                or output.get("cell_id") != request.get("cell_id") \
                or output.get("exact_output") is not True \
                or output.get("parity_verified") is not True \
                or output.get("zero_skips") is not True \
                or isinstance(output.get("skipped_count"), bool) \
                or output.get("skipped_count") != 0 \
                or output.get("payload_sha256") != hash_value(output.get("payload")) \
                or request.get("resource_claim_sha256") != claim_sha \
                or claim_handshake.get("handshake_sha256") != hash_value({
                    key: value for key, value in claim_handshake.items()
                    if key != "handshake_sha256"
                }) \
                or claim_handshake.get("request_sha256") \
                != request.get("request_sha256") \
                or claim_handshake.get("resource_claim_sha256") != claim_sha \
                or claim_handshake.get("heavy_work_started") is not False \
                or claim_ack.get("ack_sha256") != hash_value({
                    key: value for key, value in claim_ack.items()
                    if key != "ack_sha256"
                }) \
                or claim_ack.get("request_sha256") != request.get("request_sha256") \
                or claim_ack.get("resource_claim_sha256") != claim_sha \
                or claim_ack.get("target_claim_handshake_sha256") \
                != claim_handshake.get("handshake_sha256") \
                or claim_ack.get("heavy_work_authorized") is not True \
                or resource_guard.get("guard_sha256") != hash_value({
                    key: value for key, value in resource_guard.items()
                    if key != "guard_sha256"
                }) \
                or resource_guard.get("resource_claim_sha256") != claim_sha \
                or resource_guard.get("target_claim_ack_sha256") \
                != claim_ack.get("ack_sha256") \
                or resource_guard.get("rss_limit_exceeded") is not False \
                or resource_guard.get("guard_complete") is not True:
            raise ValueError("fixture output failed exact/parity/zero-skip contract")
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA, "version": VERSION,
            "validator_profile": "fixture-strict-exact-parity-zero-skip",
            "phase": request["phase"], "cell_id": request["cell_id"],
            "request_sha256": request["request_sha256"],
            "resource_claim_sha256": claim_sha,
            "target_claim_handshake_sha256": claim_handshake[
                "handshake_sha256"
            ],
            "target_claim_ack_sha256": claim_ack["ack_sha256"],
            "target_process_identity_sha256": claim_ack[
                "target_process_identity"
            ]["process_identity_sha256"],
            "target_resource_guard_sha256": resource_guard["guard_sha256"],
            "output": reference(args.output),
            "exact_output": True, "parity_verified": True,
            "zero_skips": True, "skipped_count": 0,
            "semantic_checks": [
                "schema", "phase", "cell", "payload-hash",
                "exact-output", "parity", "zero-skip",
                "resource-claim", "pre-work-handshake", "launcher-ack",
                "target-exit-identity", "tree-rss-guard",
            ],
        }
        receipt["receipt_sha256"] = hash_value(receipt)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError,
            ValueError, KeyError) as exc:
        print(f"fixture semantic validation blocked: {exc}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    raise SystemExit(main())
