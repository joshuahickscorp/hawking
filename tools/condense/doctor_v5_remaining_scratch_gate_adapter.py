#!/usr/bin/env python3.12
"""Pure, default-off future-generation adapter for remaining-scratch admission.

This module is not imported by the live Doctor queue.  It has no mutation,
activation, lock, process, signal, or resume operation.  A future reviewed
queue generation may call :func:`evaluate_gate` only with a frozen binding and
a fresh remaining-scratch receipt.  The adapter independently recomputes that
receipt from its source-bound worker request/checkpoint before granting any
credit.

Absent, stale, invalid, or drifting evidence never raises into a permissive
path: it returns the conservative 150 GB reserve + complete 48 GB scratch +
complete packed-output projection fallback.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable

import doctor_v5_remaining_scratch_ledger as ledger


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.doctor_v5_remaining_scratch_gate_decision.v1"
VERSION = "2026-07-14.1"
TARGET_CELL_ID = "qwen2-5-14b__3bpw__codec-control"
PLAN_SHA256 = "3d254b5f7fcc5f02b55f2a71f306f7f6852839b699fd14ab4ddf5a05dbaa0106"
CELL_IDENTITY_SHA256 = "8f795db9a669d6d36b14c928187478407fd1a9a1a236eab74adbbe2589d6394f"
WORKER_REQUEST_SHA256 = "a88ef27ddedcfc74af0e3e01f1f57cff5a4a7d9ee4531b0c1838ba606888a6a1"
WORKER_CHECKPOINT_SHA256 = "c69184954cc0e8b59cce9b0d1a81c9e54c860d4de70506c4441867a8ab2dbf14"
DECLARED_SCRATCH_BYTES = 48_000_000_000
DISK_RESERVE_BYTES = 150_000_000_000
PROJECTED_PACKED_OUTPUT_BYTES = 7_092_638_887
MAX_RECEIPT_AGE_SECONDS = 120.0
SHA_RE = re.compile(r"[0-9a-f]{64}")


class GateError(RuntimeError):
    """A source binding cannot authorize phase-aware credit."""


@dataclass(frozen=True)
class FrozenGateBinding:
    plan_path: Path
    plan_sha256: str
    cell_id: str
    cell_identity_sha256: str
    worker_request_path: Path
    worker_request_sha256: str
    worker_checkpoint_sha256: str
    declared_scratch_bytes: int
    disk_reserve_bytes: int
    projected_packed_output_bytes: int


PRODUCTION_BINDING = FrozenGateBinding(
    plan_path=ROOT / "reports/condense/doctor_v5_ultra/campaign_plan.json",
    plan_sha256=PLAN_SHA256, cell_id=TARGET_CELL_ID,
    cell_identity_sha256=CELL_IDENTITY_SHA256,
    worker_request_path=ROOT / (
        "reports/condense/doctor_v5_ultra/results/"
        "qwen2-5-14b__3bpw__codec-control/strand_ladder/request.json"),
    worker_request_sha256=WORKER_REQUEST_SHA256,
    worker_checkpoint_sha256=WORKER_CHECKPOINT_SHA256,
    declared_scratch_bytes=DECLARED_SCRATCH_BYTES,
    disk_reserve_bytes=DISK_RESERVE_BYTES,
    projected_packed_output_bytes=PROJECTED_PACKED_OUTPUT_BYTES,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_time(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise GateError("receipt time is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateError("receipt time is invalid") from exc
    if parsed.tzinfo is None:
        raise GateError("receipt time has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def _binding_errors(binding: FrozenGateBinding, root: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(binding, FrozenGateBinding):
        return ["gate binding type is invalid"]
    if any(not _valid_sha(value) for value in (
            binding.plan_sha256, binding.cell_identity_sha256,
            binding.worker_request_sha256, binding.worker_checkpoint_sha256)):
        errors.append("gate binding contains an invalid SHA-256")
    if binding.disk_reserve_bytes != DISK_RESERVE_BYTES:
        errors.append("gate binding reserve is not exactly 150 decimal GB")
    if binding.declared_scratch_bytes != DECLARED_SCRATCH_BYTES:
        errors.append("gate binding scratch is not exactly the frozen 48 decimal GB")
    if isinstance(binding.projected_packed_output_bytes, bool) \
            or not isinstance(binding.projected_packed_output_bytes, int) \
            or binding.projected_packed_output_bytes < 0:
        errors.append("gate binding packed projection is invalid")
    try:
        plan, plan_ref = ledger._read_json_stable(binding.plan_path, root)
        if plan_ref["sha256"] is None:  # defensive schema assertion
            errors.append("plan stable binding is absent")
        if plan.get("plan_sha256") != binding.plan_sha256 \
                or plan.get("plan_sha256") \
                != _hash_value(_without(plan, "plan_sha256")):
            errors.append("campaign plan generation changed")
        cells = plan.get("cells")
        cell = next((row for row in cells if isinstance(row, dict)
                     and row.get("cell_id") == binding.cell_id), None) \
            if isinstance(cells, list) else None
        if not isinstance(cell, dict) \
                or cell.get("cell_identity_sha256") != binding.cell_identity_sha256 \
                or cell.get("projected_output_bytes") \
                != binding.projected_packed_output_bytes \
                or cell.get("source_deletion_permitted") is not False:
            errors.append("campaign cell identity/projection/source policy changed")
    except (ledger.ScratchLedgerError, OSError, TypeError, ValueError) as exc:
        errors.append(f"campaign plan binding is unreadable: {exc}")
    return errors


def _receipt_core(receipt: dict[str, Any]) -> dict[str, Any]:
    return {name: child for name, child in receipt.items()
            if name not in {"observed_at", "receipt_sha256"}}


def _fallback(binding: FrozenGateBinding, free_bytes: int | None,
              reasons: list[str]) -> dict[str, Any]:
    required = (binding.disk_reserve_bytes + binding.declared_scratch_bytes
                + binding.projected_packed_output_bytes)
    valid_free = (isinstance(free_bytes, int) and not isinstance(free_bytes, bool)
                  and free_bytes >= 0)
    value: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION,
        "mode": "conservative-full-scratch-fallback",
        "cell_id": binding.cell_id, "plan_sha256": binding.plan_sha256,
        "phase_aware_credit_applied": False,
        "disk_reserve_bytes": binding.disk_reserve_bytes,
        "scratch_bytes_charged": binding.declared_scratch_bytes,
        "packed_output_bytes_charged": binding.projected_packed_output_bytes,
        "required_free_bytes": required, "observed_free_bytes": free_bytes,
        "capacity_ok": valid_free and free_bytes >= required,
        "fallback_reasons": sorted(set(reasons)) or ["phase-aware evidence absent"],
        "presented_receipt_consumed": False,
        "receipt_recomputed_from_frozen_sources": False,
        "caller_declared_reduction_permitted": False,
        "activation_permitted": False, "source_deletion_permitted": False,
    }
    value["decision_sha256"] = _hash_value(value)
    return value


def _evaluate_gate_for_test(
        binding: FrozenGateBinding, presented_receipt: Any, *, free_bytes: int | None,
        workspace_root: Path = ROOT,
        now: dt.datetime | None = None,
        ledger_builder: Callable[..., dict[str, Any]] = ledger.build_ledger) -> dict[str, Any]:
    """Private deterministic seam; production uses :func:`evaluate_production_gate`."""
    errors = _binding_errors(binding, workspace_root)
    if not isinstance(presented_receipt, dict):
        errors.append("remaining-scratch receipt is absent")
    else:
        errors.extend(ledger.validate_receipt(presented_receipt))
        try:
            observed_now = now or _now()
            if not isinstance(observed_now, dt.datetime) or observed_now.tzinfo is None:
                raise GateError("current time is naive or invalid")
            age = (observed_now.astimezone(dt.timezone.utc)
                   - _parse_time(presented_receipt.get("observed_at"))) \
                .total_seconds()
            if age < -5 or age > MAX_RECEIPT_AGE_SECONDS:
                errors.append("remaining-scratch receipt is stale or future-dated")
        except (GateError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    recomputed: dict[str, Any] | None = None
    if not errors:
        try:
            recomputed = ledger_builder(
                binding.worker_request_path,
                projected_packed_output_bytes=binding.projected_packed_output_bytes,
                workspace_root=workspace_root,
            )
            errors.extend(ledger.validate_receipt(recomputed))
        except (ledger.ScratchLedgerError, OSError, TypeError, ValueError) as exc:
            errors.append(f"remaining-scratch recomputation failed: {exc}")
    if not errors and isinstance(recomputed, dict):
        if presented_receipt.get("request", {}).get("sha256") \
                != binding.worker_request_sha256 \
                or presented_receipt.get("checkpoint", {}).get("sha256") \
                != binding.worker_checkpoint_sha256:
            errors.append("presented receipt differs from frozen request/checkpoint")
        if recomputed.get("request", {}).get("sha256") \
                != binding.worker_request_sha256 \
                or recomputed.get("checkpoint", {}).get("sha256") \
                != binding.worker_checkpoint_sha256:
            errors.append("recomputed receipt differs from frozen request/checkpoint")
        if _receipt_core(presented_receipt) != _receipt_core(recomputed):
            errors.append("presented receipt differs from fresh source recomputation")
        if recomputed.get("disk_reserve_bytes") != binding.disk_reserve_bytes \
                or recomputed.get("declared_total_scratch_bytes") \
                != binding.declared_scratch_bytes \
                or recomputed.get("projected_whole_packed_output_bytes") \
                != binding.projected_packed_output_bytes:
            errors.append("recomputed receipt differs from frozen byte envelopes")
    if errors or not isinstance(recomputed, dict):
        return _fallback(binding, free_bytes, errors)
    valid_free = (isinstance(free_bytes, int) and not isinstance(free_bytes, bool)
                  and free_bytes >= 0)
    required = recomputed["required_free_bytes"]
    decision: dict[str, Any] = {
        "schema": SCHEMA, "version": VERSION,
        "mode": "validated-phase-aware-remaining-scratch",
        "cell_id": binding.cell_id, "plan_sha256": binding.plan_sha256,
        "phase_aware_credit_applied": True,
        "disk_reserve_bytes": recomputed["disk_reserve_bytes"],
        "scratch_bytes_charged": recomputed["remaining_scratch_bytes"],
        "packed_output_bytes_charged": recomputed[
            "projected_remaining_packed_output_bytes"],
        "required_free_bytes": required, "observed_free_bytes": free_bytes,
        "capacity_ok": valid_free and free_bytes >= required,
        "fallback_reasons": [], "presented_receipt_consumed": True,
        "receipt_recomputed_from_frozen_sources": True,
        "presented_receipt_sha256": presented_receipt["receipt_sha256"],
        "recomputed_receipt_sha256": recomputed["receipt_sha256"],
        "request_file_sha256": binding.worker_request_sha256,
        "checkpoint_file_sha256": binding.worker_checkpoint_sha256,
        "caller_declared_reduction_permitted": False,
        "activation_permitted": False, "source_deletion_permitted": False,
    }
    decision["decision_sha256"] = _hash_value(decision)
    return decision


def _direct_free_bytes(root: Path) -> int | None:
    try:
        snapshot = os.statvfs(root)
        value = snapshot.f_bavail * snapshot.f_frsize
    except OSError:
        return None
    return value if isinstance(value, int) and value >= 0 else None


def evaluate_production_gate(presented_receipt: Any) -> dict[str, Any]:
    """Only production-facing entry: fixed sources, direct disk probe, real time."""
    return _evaluate_gate_for_test(
        PRODUCTION_BINDING, presented_receipt,
        free_bytes=_direct_free_bytes(ROOT), workspace_root=ROOT,
        now=_now(), ledger_builder=ledger.build_ledger,
    )


def validate_decision(decision: Any, binding: FrozenGateBinding) -> list[str]:
    if not isinstance(decision, dict) or decision.get("schema") != SCHEMA \
            or decision.get("version") != VERSION:
        return ["gate decision schema/version mismatch"]
    errors: list[str] = []
    if decision.get("decision_sha256") != _hash_value(
            _without(decision, "decision_sha256")):
        errors.append("gate decision self-hash mismatch")
    if decision.get("cell_id") != binding.cell_id \
            or decision.get("plan_sha256") != binding.plan_sha256:
        errors.append("gate decision source binding changed")
    if decision.get("activation_permitted") is not False \
            or decision.get("source_deletion_permitted") is not False \
            or decision.get("caller_declared_reduction_permitted") is not False:
        errors.append("gate decision weakens isolation")
    mode = decision.get("mode")
    if mode == "conservative-full-scratch-fallback":
        expected_keys = {
            "schema", "version", "mode", "cell_id", "plan_sha256",
            "phase_aware_credit_applied", "disk_reserve_bytes",
            "scratch_bytes_charged", "packed_output_bytes_charged",
            "required_free_bytes", "observed_free_bytes", "capacity_ok",
            "fallback_reasons", "presented_receipt_consumed",
            "receipt_recomputed_from_frozen_sources",
            "caller_declared_reduction_permitted", "activation_permitted",
            "source_deletion_permitted", "decision_sha256",
        }
        if set(decision) != expected_keys:
            errors.append("fallback decision keys are not exact")
        expected = (binding.disk_reserve_bytes + binding.declared_scratch_bytes
                    + binding.projected_packed_output_bytes)
        if decision.get("phase_aware_credit_applied") is not False \
                or decision.get("scratch_bytes_charged") \
                != binding.declared_scratch_bytes \
                or decision.get("packed_output_bytes_charged") \
                != binding.projected_packed_output_bytes \
                or decision.get("required_free_bytes") != expected:
            errors.append("fallback decision is not the complete conservative envelope")
    elif mode == "validated-phase-aware-remaining-scratch":
        expected_keys = {
            "schema", "version", "mode", "cell_id", "plan_sha256",
            "phase_aware_credit_applied", "disk_reserve_bytes",
            "scratch_bytes_charged", "packed_output_bytes_charged",
            "required_free_bytes", "observed_free_bytes", "capacity_ok",
            "fallback_reasons", "presented_receipt_consumed",
            "receipt_recomputed_from_frozen_sources",
            "presented_receipt_sha256", "recomputed_receipt_sha256",
            "request_file_sha256", "checkpoint_file_sha256",
            "caller_declared_reduction_permitted", "activation_permitted",
            "source_deletion_permitted", "decision_sha256",
        }
        if set(decision) != expected_keys:
            errors.append("phase-aware decision keys are not exact")
        scratch = decision.get("scratch_bytes_charged")
        packed = decision.get("packed_output_bytes_charged")
        if decision.get("phase_aware_credit_applied") is not True \
                or decision.get("presented_receipt_consumed") is not True \
                or decision.get("receipt_recomputed_from_frozen_sources") is not True \
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in (scratch, packed)) \
                or decision.get("required_free_bytes") \
                != binding.disk_reserve_bytes + scratch + packed:
            errors.append("phase-aware decision equation or evidence contract changed")
    else:
        errors.append("gate decision mode is invalid")
    free = decision.get("observed_free_bytes")
    expected_ok = (isinstance(free, int) and not isinstance(free, bool) and free >= 0
                   and free >= decision.get("required_free_bytes", math.inf))
    if decision.get("capacity_ok") is not expected_ok:
        errors.append("gate decision capacity result differs from byte comparison")
    return errors


def _production_status() -> dict[str, Any]:
    # No receipt argument is accepted by the CLI.  Status demonstrates the
    # conservative absent-evidence behavior and cannot authorize production.
    return evaluate_production_gate(None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", nargs="?")
    args = parser.parse_args(argv)
    if args.status not in {None, "status"}:
        parser.error("only read-only status is supported")
    value = _production_status()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if not validate_decision(value, PRODUCTION_BINDING) else 2


if __name__ == "__main__":
    raise SystemExit(main())
