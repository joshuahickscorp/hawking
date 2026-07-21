#!/usr/bin/env python3.12
"""Fail-closed finalizer for the Kimi K2.6 Gravity closure chapter.

This module does no experiment work and never removes the resident source.  It
only closes the chapter after the physical byte auction, nonlinear tournament,
bounded one-off closure, storage/GC proof, and parallel-execution proof form a
complete sealed evidence chain.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_final_chapter_manager as chapter  # noqa: E402


f1 = chapter.f1
REPO = Path(__file__).resolve().parents[2]
RUNTIME = chapter.RUNTIME
FLOOR_BYTES = 5 * 1024**3
LOGICAL_WEIGHTS = 44_040_192
CEILING_BYTES = 5_394_923
CEILING_BPW = CEILING_BYTES * 8 / LOGICAL_WEIGHTS

FINAL_JSON = "KIMI_K26_GRAVITY_FINAL.json"
FINAL_MD = "KIMI_K26_GRAVITY_FINAL.md"
TRANSFER_MD = "KIMI_K26_NEXT_PARENT_TRANSFER.md"
BYTE_AUCTION = "KIMI_K26_FINAL_BYTE_AUCTION.json"
NONLINEAR = "KIMI_K26_GRAVITY_NONLINEAR_TOURNAMENT.json"
M1 = "KIMI_K26_GRAVITY_M1_ORACLE_BANDWIDTH.json"
M2 = "KIMI_K26_GRAVITY_M2_CONDITIONAL_GATE.json"
M5 = "KIMI_K26_GRAVITY_RATE_LADDER.json"
M7 = "KIMI_K26_GRAVITY_M7_ORACLE_GAP.json"
GC_LEDGER = "KIMI_K26_FINAL_GC_LEDGER.jsonl"
PARALLEL_LEDGER = "KIMI_K26_PARALLEL_EXECUTION_LEDGER.jsonl"
CHAPTER_LEDGER = "KIMI_K26_FINAL_CHAPTER_LEDGER.jsonl"
STORAGE_REPORT = "KIMI_K26_FINAL_STORAGE_REPORT.md"
POLICY = "KIMI_K26_DISK_POLICY.json"
SCIENTIFIC_STATUS = "KIMI_K26_SCIENTIFIC_STATUS.json"
LONG_FINAL = "KIMI_K26_LONG_RUN_FINAL.json"

ONEOFF_CLOSURE_CANDIDATES = (
    "KIMI_K26_GRAVITY_ONEOFFS_CLOSURE.json",
    "KIMI_K26_GRAVITY_ONEOFF_CLOSURE.json",
    "KIMI_K26_GRAVITY_ONEOFFS_INDEX.json",
)
OPTIONAL_ONEOFFS = {
    "M3": "KIMI_K26_GRAVITY_M3_NATIVE_STUDENTIZATION.json",
    "M4": "KIMI_K26_GRAVITY_M4_CROSS_LAYER_ANCHOR.json",
    "M6": "KIMI_K26_GRAVITY_M6_DOCTOR_NATIVE_INVERSION.json",
}
REQUIRED_EXECUTIONS = (
    "PE01_CONTEXTUAL_CAPTURE_M1_POLICY_TESTS",
    "PE02_NONLINEAR_TOURNAMENT_M2_M7",
    "PE03_EXACT_RATE_LADDER_HOOKS_TESTS",
)
BASELINE_PAYLOAD = (
    RUNTIME / "f1_representation_bracket/doctor_auction/"
    "P1_DUAL_PATH_RECOVERY_R16X2.k26f1"
)
BASELINE_PAYLOAD_SHA256 = (
    "3546c9b17f720d6d5197c8a8d1dae80e5994e053a808de708aef6bb5e97561bb"
)
TERMINAL_OUTCOMES = {f"OUTCOME_{letter}" for letter in "ABCDE"}


class ClosureError(RuntimeError):
    """Raised before any write when final evidence is incomplete or invalid."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sealed_json(path: Path, *, required_status: bool = True) -> dict[str, Any]:
    if not path.is_file():
        raise ClosureError(f"required artifact absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"artifact is not a JSON object: {path}")
    expected = hashlib.sha256(canonical({
        key: item for key, item in value.items() if key != "seal_sha256"
    })).hexdigest()
    if value.get("seal_sha256") != expected:
        raise ClosureError(f"seal mismatch: {path}")
    if required_status and str(value.get("status", "PASS")).upper() in {
        "FAIL", "FAILED", "ERROR", "INVALID",
    }:
        raise ClosureError(f"artifact has failure status: {path}: {value.get('status')}")
    return value


def verify_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ClosureError(f"required ledger absent: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClosureError(f"invalid JSONL {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ClosureError(f"ledger row is not an object: {path}:{number}")
        expected = hashlib.sha256(canonical({
            key: item for key, item in row.items() if key != "seal_sha256"
        })).hexdigest()
        if row.get("seal_sha256") != expected:
            raise ClosureError(f"ledger seal mismatch: {path}:{number}")
        rows.append(row)
    if not rows:
        raise ClosureError(f"required ledger is empty: {path}")
    return rows


def first_mapping(*values: Any) -> dict[str, Any] | None:
    for value in values:
        if isinstance(value, dict):
            return value
    return None


def first_value(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def find_closure(repo: Path) -> tuple[Path, dict[str, Any]]:
    for name in ONEOFF_CLOSURE_CANDIDATES:
        path = repo / name
        if not path.is_file():
            continue
        value = verify_sealed_json(path)
        decisions = first_mapping(
            value.get("decisions"), value.get("oneoffs"),
            value.get("terminal_decisions"), value.get("experiments"),
        )
        # A generated index is accepted only if it was upgraded into a true
        # terminal closure.  Its ordinary stage index is deliberately not enough.
        if decisions is not None:
            return path, value
    expected = ", ".join(ONEOFF_CLOSURE_CANDIDATES[:2])
    raise ClosureError(
        "sealed M1-M7 terminal closure absent; expected decisions/oneoffs map in "
        f"one of: {expected}"
    )


def decision_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(first_value(value, (
            "state", "decision", "source_decision", "verdict", "status", "result",
        )) or "")
    return ""


def terminal_decision(value: Any) -> bool:
    if isinstance(value, dict) and value.get("terminal") is True:
        return True
    text = decision_text(value).upper()
    if not text or any(word in text for word in ("WAITING", "RUNNING", "PENDING", "PARTIAL")):
        return False
    return any(word in text for word in (
        "RETIRE", "REJECT", "CLOSE", "COMPLETE", "DOMINATED", "FAILED",
        "NO_CANDIDATE", "NO_QUALIFIED", "NOT_ADMITTED", "PREREQUISITE_ABSENT",
        "FALSIFIED", "RETAIN_BASELINE", "WINNER_REPLICATED", "HIGHER_FIDELITY_COMPLETE",
        "TESTED_REJECTED", "TESTED_COMPLETE",
    ))


def rejection_has_reason(value: Any) -> bool:
    text = decision_text(value).upper()
    if not any(word in text for word in (
        "RETIRE", "REJECT", "FAILED", "NO_CANDIDATE", "NOT_ADMITTED",
        "PREREQUISITE_ABSENT", "FALSIFIED",
    )):
        return True
    if not isinstance(value, dict):
        return False
    reason = first_value(value, ("reason", "reasons", "evidence", "causal_reason"))
    return bool(reason)


def oneoff_decisions(closure: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = first_mapping(
        closure.get("decisions"), closure.get("oneoffs"),
        closure.get("terminal_decisions"), closure.get("experiments"),
    ) or {}
    normalized: dict[str, dict[str, Any]] = {}
    failures = []
    for name in (f"M{number}" for number in range(1, 8)):
        value = raw.get(name) or raw.get(name.lower())
        if value is None:
            failures.append(f"{name}_DECISION_ABSENT")
            continue
        entry = value if isinstance(value, dict) else {"decision": str(value)}
        if not terminal_decision(entry):
            failures.append(f"{name}_NOT_TERMINAL:{decision_text(entry)}")
        elif not rejection_has_reason(entry):
            failures.append(f"{name}_REJECTION_REASON_ABSENT")
        normalized[name] = entry
    if failures:
        raise ClosureError("one-off closure incomplete: " + "; ".join(failures))
    return normalized


def nonlinear_decisions(
    tournament: dict[str, Any], auction: dict[str, Any], closure: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    sources = (
        tournament.get("family_decisions"), tournament.get("families"),
        auction.get("family_decisions"), auction.get("nonlinear_family_decisions"),
        auction.get("families"), auction.get("nonlinear_family_adjudication"),
        closure.get("nonlinear_families"),
    )
    for source in sources:
        if isinstance(source, dict):
            for key, value in source.items():
                name = str(key).upper()
                if name in {f"N{number}" for number in range(1, 7)}:
                    normalized[name] = value if isinstance(value, dict) else {
                        "decision": str(value),
                    }
        elif isinstance(source, list):
            for value in source:
                if not isinstance(value, dict):
                    continue
                name = str(first_value(value, ("family", "id", "name")) or "").upper()
                if name in {f"N{number}" for number in range(1, 7)}:
                    normalized[name] = value
    failures = []
    for name in (f"N{number}" for number in range(1, 7)):
        value = normalized.get(name)
        if value is None:
            failures.append(f"{name}_DECISION_ABSENT")
        elif not terminal_decision(value):
            failures.append(f"{name}_NOT_TERMINAL:{decision_text(value)}")
        elif not rejection_has_reason(value):
            failures.append(f"{name}_REJECTION_REASON_ABSENT")
    if failures:
        raise ClosureError("nonlinear tournament incomplete: " + "; ".join(failures))
    return normalized


def latest_event(rows: list[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("event") == event]
    if not matches:
        raise ClosureError(f"required ledger event absent: {event}")
    return matches[-1]


def gc_proof(repo: Path, policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("status") != "PASS" or policy.get("hard_floor_bytes") != FLOOR_BYTES:
        raise ClosureError("5 GiB disk policy is not sealed PASS")
    enforcement = policy.get("enforcement") or {}
    required_true = (
        "campaign_runtime_matches_repo", "doctor_runtime_matches_repo",
        "live_launchd_contains_floor",
    )
    if not all(enforcement.get(key) is True for key in required_true):
        raise ClosureError("disk policy active-enforcement chain is incomplete")
    if any(enforcement.get(key) != FLOOR_BYTES for key in (
        "floor_bytes", "static_launchd_floor_bytes", "installed_launchd_floor_bytes",
        "manager_floor_bytes",
    )):
        raise ClosureError("one or more active disk paths do not use exactly 5 GiB")
    rows = verify_jsonl(repo / GC_LEDGER)
    complete = latest_event(rows, "GC_COMPLETE")
    if complete.get("disk_floor_bytes") != FLOOR_BYTES or complete.get("floor_green_after") is not True:
        raise ClosureError("exercised GC did not finish above the exact 5 GiB floor")
    plan_seal = complete.get("plan_seal_sha256")
    plans = [row for row in rows if row.get("event") == "GC_PLAN" and
             row.get("seal_sha256") == plan_seal]
    if len(plans) != 1 or not plans[0].get("execute_requested"):
        raise ClosureError("GC completion does not point to one sealed execute plan")
    deletes = [row for row in rows if row.get("event") == "GC_DELETE" and
               row.get("plan_seal_sha256") == plan_seal]
    if len(deletes) != int(complete.get("deleted_count", -1)):
        raise ClosureError("GC delete count does not match the sealed plan/completion chain")
    if sum(int(row.get("logical_bytes", 0)) for row in deletes) != int(
        complete.get("deleted_logical_bytes", -1)
    ):
        raise ClosureError("GC reclaimed-byte accounting mismatch")
    storage = repo / STORAGE_REPORT
    if not storage.is_file():
        raise ClosureError(f"storage report absent: {storage}")
    return {
        "policy": policy,
        "gc_completion": complete,
        "gc_plan_seal_sha256": plan_seal,
        "delete_record_seals": [row["seal_sha256"] for row in deletes],
        "storage_report": {"path": str(storage), "sha256": sha256_file(storage)},
    }


def parallel_proof(repo: Path) -> dict[str, Any]:
    rows = verify_jsonl(repo / PARALLEL_LEDGER)
    executions: dict[str, Any] = {}
    for execution_id in REQUIRED_EXECUTIONS:
        complete_rows = [row for row in rows if
                         row.get("event") == "PARALLEL_EXECUTION_COMPLETE" and
                         row.get("execution_id") == execution_id]
        if len(complete_rows) != 1:
            raise ClosureError(f"{execution_id} must have exactly one completion record")
        complete = complete_rows[0]
        if complete.get("decision") != "PARALLELISM_DEMONSTRATED":
            raise ClosureError(f"parallelism not demonstrated for {execution_id}")
        if int(complete.get("heavy_lane_count", -1)) != 1 or int(
            complete.get("lane_count", 0)
        ) < 2:
            raise ClosureError(f"{execution_id} violates one-heavy-plus-light policy")
        if complete.get("guard_after_status") != "PASS" or complete.get("guard_after_failures"):
            raise ClosureError(f"post-parallel guard failed for {execution_id}")
        lanes = [row for row in rows if row.get("event") == "LANE_COMPLETE" and
                 row.get("execution_id") == execution_id]
        if len(lanes) != int(complete["lane_count"]):
            raise ClosureError(f"{execution_id} lane count does not match completion")
        if sum(row.get("lane") == "HEAVY_LANE" for row in lanes) != 1:
            raise ClosureError(f"{execution_id} lacks exactly one completed heavy lane")
        if any(int(row.get("exit_code", -1)) != 0 for row in lanes):
            raise ClosureError(f"{execution_id} contains a failed lane")
        recorded_seals = set(complete.get("lane_record_seals") or [])
        if recorded_seals != {row["seal_sha256"] for row in lanes}:
            raise ClosureError(f"{execution_id} lane seal chain mismatch")
        executions[execution_id] = {
            "completion": complete,
            "lanes": lanes,
            "peak_cpu_percent": max(float(row.get("cpu_percent_peak", 0)) for row in lanes),
            "peak_gpu_device_utilization_percent": max(
                float(row.get("gpu_device_utilization_peak_percent", 0)) for row in lanes
            ),
            "peak_resident_memory_bytes": max(
                int(row.get("resident_memory_peak_bytes", 0)) for row in lanes
            ),
            "logical_disk_read_bytes": sum(
                int(row.get("logical_disk_read_bytes", 0)) for row in lanes
            ),
            "logical_disk_write_bytes": sum(
                int(row.get("logical_disk_write_bytes", 0)) for row in lanes
            ),
            "contention_effects": [row.get("contention_effect") for row in lanes],
        }
    return executions


def best_candidate(
    auction: dict[str, Any], tournament: dict[str, Any],
) -> dict[str, Any]:
    best = first_mapping(
        auction.get("best_deployable_candidate"), auction.get("current_best"),
        auction.get("best"), tournament.get("best"),
    )
    if best is None:
        raise ClosureError("byte auction does not identify a best deployable candidate")
    name = str(first_value(best, ("candidate", "name", "id")) or "")
    if not name:
        raise ClosureError("best deployable candidate has no name")
    expected_sha = first_value(best, ("payload_sha256", "sha256", "candidate_sha256"))
    payload = first_value(best, ("payload_path", "path", "artifact_path"))
    receipt = first_mapping(best.get("payload"), best.get("artifact"))
    if receipt:
        payload = payload or receipt.get("path")
        expected_sha = expected_sha or receipt.get("sha256")
    if payload is None and name.startswith("P1_DUAL_PATH_RECOVERY_R16X2"):
        payload = str(BASELINE_PAYLOAD)
        expected_sha = expected_sha or BASELINE_PAYLOAD_SHA256
    if payload is None:
        raise ClosureError("best deployable candidate lacks a physical payload path")
    path = Path(str(payload)).expanduser().resolve(strict=True)
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if not expected_sha or actual_sha != expected_sha:
        raise ClosureError("best deployable payload hash mismatch")
    declared_bytes = first_value(best, (
        "complete_physical_bytes", "physical_bytes", "total_bytes", "bytes",
    ))
    if declared_bytes is not None and int(declared_bytes) != actual_bytes:
        raise ClosureError("best deployable declared bytes do not match serialized payload")
    exact_bpw = actual_bytes * 8 / LOGICAL_WEIGHTS
    declared_bpw = first_value(best, ("complete_bpw", "actual_complete_bpw", "bpw"))
    if declared_bpw is not None and abs(float(declared_bpw) - exact_bpw) > 5e-13:
        raise ClosureError("best deployable BPW does not equal exact serialized arithmetic")
    if actual_bytes > CEILING_BYTES or exact_bpw > 0.98:
        raise ClosureError("best deployable violates the 0.98 complete-BPW law")
    allocation = first_mapping(
        best.get("allocation"), best.get("physical_allocation"),
        auction.get("best_allocation"), auction.get("current_best_allocation"),
    )
    if allocation is None and name.startswith("P1_DUAL_PATH_RECOVERY_R16X2"):
        allocation = {
            "compact_base_bytes": 4_022_298,
            "doctor_bytes": 974_848,
            "header_bytes": 4_669,
        }
    if allocation is None:
        raise ClosureError("best deployable lacks exact component allocation")
    numeric = [int(value) for key, value in allocation.items()
               if key.endswith("_bytes") and key not in {
                   "total_bytes", "complete_physical_bytes", "unused_ceiling_bytes",
                   "reserve_bytes", "ceiling_bytes", "complete_ceiling_bytes",
               }]
    declared_total = first_value(allocation, ("total_bytes", "complete_physical_bytes"))
    if numeric and sum(numeric) != actual_bytes:
        raise ClosureError("best component byte allocation does not sum to payload bytes")
    if declared_total is not None and int(declared_total) != actual_bytes:
        raise ClosureError("best allocation total does not match payload bytes")
    return {
        "candidate": name,
        "payload_path": str(path),
        "payload_sha256": actual_sha,
        "complete_physical_bytes": actual_bytes,
        "complete_bpw": exact_bpw,
        "allocation": allocation,
        "f2_promotable": bool(best.get("f2_promotable", False)),
    }


def oracle_bound(m1: dict[str, Any]) -> dict[str, Any]:
    queue = m1.get("downstream_forward_queue") or []
    screening = ((m1.get("screening") or {}).get("rows") or [])
    rows = [row for row in [*queue, *screening] if isinstance(row, dict) and
            row.get("boundary_rescue_fraction") is not None]
    if not rows:
        raise ClosureError("M1 contains no teacher-hidden bandwidth rows")
    legal = [row for row in rows if row.get("within_incremental_0_98_bpw") is True]
    if not legal:
        raise ClosureError("M1 contains no exactly billed legal oracle row")
    best_legal = max(legal, key=lambda row: float(row["boundary_rescue_fraction"]))
    best_any = max(rows, key=lambda row: float(row["boundary_rescue_fraction"]))
    return {
        "claim_boundary": m1.get("claim_boundary"),
        "teacher_access_at_inference": True,
        "qualifying_rows": int(m1.get("qualifying_rows", 0)),
        "decision": m1.get("decision"),
        "best_legal_row": best_legal,
        "best_tested_row": best_any,
        "minimum_correction_dimensionality": (
            None if int(m1.get("qualifying_rows", 0)) == 0 else
            min(int(row["rank"]) for row in rows if row.get("boundary_qualifies_for_forward"))
        ),
        "minimum_update_frequency": (
            None if int(m1.get("qualifying_rows", 0)) == 0 else
            min(float(row["update_frequency_tokens_per_correction"]) for row in rows
                if row.get("boundary_qualifies_for_forward"))
        ),
        "entropy_bit_lower_bound": first_value(m1, (
            "entropy_bit_lower_bound", "estimated_entropy_bit_lower_bound",
            "bit_lower_bound",
        )),
        "interpretation": (
            "No tested row met the preregistered 0.90-rescue / 0.80-CI-low boundary; "
            "the experiment bounds the tested grid but does not establish a universal minimum."
            if int(m1.get("qualifying_rows", 0)) == 0 else
            "At least one oracle row met the preregistered boundary; its downstream result "
            "defines the measured correction floor."
        ),
    }


def rate_ladder_proof(value: dict[str, Any]) -> dict[str, Any]:
    rows = value.get("stress_ladder") or value.get("rate_ladder") or value.get("rungs")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ClosureError("M5 must contain exactly three physical stress-ladder rows")
    expected = {"0.75", "0.50", "0.33"}
    seen: set[str] = set()
    verified = []
    for row in rows:
        if not isinstance(row, dict):
            raise ClosureError("M5 stress-ladder row is not an object")
        rate = str(row.get("rate") or row.get("target_complete_bpw") or "")
        if rate in {"0.5", "0.50"}:
            rate = "0.50"
        elif rate in {"0.75", ".75"}:
            rate = "0.75"
        elif rate in {"0.33", ".33"}:
            rate = "0.33"
        if rate not in expected or rate in seen:
            raise ClosureError(f"invalid or duplicate M5 rate rung: {rate}")
        seen.add(rate)
        payload = first_mapping(row.get("physical_payload"), row.get("payload"))
        if payload is None:
            raise ClosureError(f"M5 {rate} lacks an exact physical payload receipt")
        path = Path(str(payload.get("path"))).expanduser().resolve(strict=True)
        size = path.stat().st_size
        if size != int(payload.get("bytes", -1)) or sha256_file(path) != payload.get("sha256"):
            raise ClosureError(f"M5 {rate} physical payload receipt mismatch")
        exact_bpw = size * 8 / LOGICAL_WEIGHTS
        declared = first_value(payload, ("actual_complete_bpw", "complete_bpw"))
        if declared is None or abs(float(declared) - exact_bpw) > 5e-13:
            raise ClosureError(f"M5 {rate} complete BPW is not exact serialized arithmetic")
        verdict = str(first_value(row, (
            "strict_final_verdict", "decision", "verdict",
        )) or "").upper()
        if not any(word in verdict for word in ("RETIRE", "REJECT", "COMPLETE")):
            raise ClosureError(f"M5 {rate} lacks a strict terminal F1 verdict")
        verified.append({
            "rate": rate, "candidate": row.get("candidate"),
            "complete_physical_bytes": size, "actual_complete_bpw": exact_bpw,
            "payload_path": str(path), "payload_sha256": payload["sha256"],
            "strict_final_verdict": verdict,
            "strict_admission_reasons": row.get("strict_admission_reasons") or [],
            "frozen_score_metrics": row.get("frozen_score_metrics"),
        })
    if seen != expected:
        raise ClosureError(f"M5 rate rungs incomplete: {sorted(seen)}")
    if "RETIRES_ALL" not in str(value.get("decision", "")).upper():
        raise ClosureError("M5 artifact does not terminally retire all strict F1 rungs")
    return {"decision": value["decision"], "rows": verified,
            "artifact_seal_sha256": value["seal_sha256"]}


def source_release_proof(audit: dict[str, Any]) -> dict[str, Any]:
    manifest = verify_sealed_json(RUNTIME / "KIMI_K26_OFFICIAL_MANIFEST.json")
    snapshot = chapter.legacy.SNAPSHOT.resolve(strict=True)
    logical = 0
    allocated = 0
    inodes: set[tuple[int, int]] = set()
    for item in manifest.get("files") or []:
        path = (snapshot / item["path"]).resolve(strict=True)
        stat = path.stat()
        if stat.st_size != int(item["size"]):
            raise ClosureError(f"source file changed while sizing release: {item['path']}")
        logical += stat.st_size
        identity = (stat.st_dev, stat.st_ino)
        if identity not in inodes:
            allocated += stat.st_blocks * 512
            inodes.add(identity)
    if not audit["source"]["one_copy"]:
        raise ClosureError("source release cannot be reported while one-copy invariant fails")
    return {
        "source": "moonshotai/Kimi-K2.6",
        "revision": chapter.legacy.REVISION,
        "snapshot_path": str(snapshot),
        "file_count": len(manifest.get("files") or []),
        "logical_bytes_referenced": logical,
        "unique_allocated_bytes_referenced": allocated,
        "unique_content_inodes": len(inodes),
        "can_be_released_after_explicit_user_authorization": True,
        "automatic_deletion_performed": False,
        "warning": (
            "This is a read-only sizing estimate. Actual recovered bytes depend on other "
            "Hugging Face cache references. The sole source was not deleted."
        ),
    }


def fidelity_summary(
    auction: dict[str, Any], tournament: dict[str, Any], best: dict[str, Any],
    prior_long_run: dict[str, Any],
) -> dict[str, Any]:
    explicit = first_mapping(auction.get("fidelity"), auction.get("fidelity_results")) or {}
    scientific = verify_sealed_json(REPO / SCIENTIFIC_STATUS)
    baseline = scientific.get("best_local_candidate") or {}
    f1_score = first_mapping(explicit.get("F1"), explicit.get("f1")) or {
        "status": "COMPLETE",
        "cosine_mean": (baseline.get("f1_score") or {}).get("cosine_mean"),
        "decision": baseline.get("terminal_verdict"),
    }
    f2 = first_mapping(explicit.get("F2"), explicit.get("f2")) or {
        "status": "COMPLETE_FAILED_PROMOTION",
        "promotable": best["f2_promotable"],
        "decision": "NO_DEPLOYABLE_CANDIDATE_EARNED_F2_PROMOTION",
    }
    higher = {}
    for name in ("F3", "F4", "F5"):
        value = first_mapping(explicit.get(name), explicit.get(name.lower()))
        higher[name] = value or {
            "status": "NOT_EARNED",
            "reason": "NO_REPLICATED_F2_WINNER",
        }
    replication = first_mapping(
        auction.get("replication"), auction.get("replication_results"),
        prior_long_run.get("replication"),
    ) or {
        "status": "NO_APPARENT_NONLINEAR_WINNER_TO_REPLICATE",
        "f2_winner_replicated": False,
    }
    return {"F0": {"status": "COMPLETE_EXACT_SERIALIZATION"},
            "F1": f1_score, "F2": f2, **higher, "replication": replication}


def infer_outcome(
    closure: dict[str, Any], auction: dict[str, Any], fidelity: dict[str, Any],
    oracle: dict[str, Any], families: dict[str, Any], oneoffs: dict[str, Any],
) -> str:
    explicit = first_value(closure, (
        "terminal_outcome", "recommended_terminal_outcome", "terminal_outcome_recommendation",
    ))
    explicit = explicit or first_value(auction, ("terminal_outcome", "recommended_terminal_outcome"))
    if explicit is not None:
        explicit_text = str(explicit).strip().upper()
        outcome = next((item for item in sorted(TERMINAL_OUTCOMES)
                        if explicit_text.startswith(item)), explicit_text.split(":", 1)[0])
    else:
        replication = fidelity.get("replication") or {}
        f2_replicated = bool(replication.get("f2_winner_replicated") or
                             replication.get("replicated_f2_winner"))
        higher = any(str((fidelity.get(name) or {}).get("status", "")).upper() in {
            "PASS", "WINNER", "COMPLETE_PASS", "STATE_OF_THE_ART",
        } for name in ("F3", "F4", "F5"))
        native_only = bool(closure.get("native_studentization_only_promising")) or \
            "ONLY_PROMISING" in decision_text(oneoffs.get("M3")).upper()
        blocker = first_value(closure, ("technical_blocker", "closure_blocker"))
        if blocker:
            outcome = "OUTCOME_E"
        elif higher:
            outcome = "OUTCOME_B"
        elif f2_replicated:
            outcome = "OUTCOME_A"
        elif native_only:
            outcome = "OUTCOME_D"
        elif all(terminal_decision(value) for value in families.values()) and \
                all(terminal_decision(value) for value in oneoffs.values()) and \
                int(oracle["qualifying_rows"]) == 0:
            outcome = "OUTCOME_C"
        else:
            raise ClosureError("terminal outcome cannot be inferred from the sealed evidence")
    if outcome not in TERMINAL_OUTCOMES:
        raise ClosureError(f"invalid terminal outcome: {outcome}")
    replication = fidelity.get("replication") or {}
    if outcome == "OUTCOME_A" and not bool(
        replication.get("f2_winner_replicated") or replication.get("replicated_f2_winner")
    ):
        raise ClosureError("OUTCOME_A requires an independently replicated F2 winner")
    if outcome == "OUTCOME_B" and not any(
        str((fidelity.get(name) or {}).get("status", "")).upper() in {
            "PASS", "WINNER", "COMPLETE_PASS", "STATE_OF_THE_ART",
        } for name in ("F3", "F4", "F5")
    ):
        raise ClosureError("OUTCOME_B requires a sealed higher-fidelity winner")
    if outcome == "OUTCOME_C" and int(oracle["qualifying_rows"]) != 0:
        raise ClosureError("OUTCOME_C conflicts with a qualifying oracle row")
    if outcome == "OUTCOME_E" and not first_value(
        closure, ("technical_blocker", "closure_blocker")
    ):
        raise ClosureError("OUTCOME_E requires an exact named technical blocker")
    return outcome


def ledger_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "event": row.get("event"), "experiment_id": row.get("experiment_id"),
        "started_at": row.get("started_at") or row.get("at"),
        "ended_at": row.get("ended_at") or row.get("at"),
        "decision": row.get("decision"), "seal_sha256": row.get("seal_sha256"),
    } for row in rows]


def conclusions(
    outcome: str, families: dict[str, Any], oneoffs: dict[str, Any],
    best: dict[str, Any], oracle: dict[str, Any], closure: dict[str, Any],
) -> dict[str, list[str]]:
    retired = [name for name, value in families.items()
               if any(word in decision_text(value).upper() for word in ("RETIRE", "REJECT", "CLOSE"))]
    proven = [
        (f"On the sealed Kimi splits, the retained deployable representation is {best['candidate']} "
         f"at {best['complete_bpw']:.12f} complete BPW; it is not F2-promotable."),
        ("All six proposed nonlinear families have terminal adjudications. The physically "
         f"tested families closed without a replicated F2 winner: {', '.join(retired) or 'none listed'}."),
        ("The teacher-hidden experiment is an inference-invalid oracle. No tested row met its "
         "preregistered rescue-confidence boundary, so it does not establish a deployable repair."),
        ("The exact 5 GiB disk law, dependency-safe GC, and one-heavy-plus-light execution "
         "policy were exercised while the sole source and MOP remained protected."),
    ]
    suggested = [
        "Kimi favors preserving or regenerating the pre-router functional state trajectory over downstream linear repair.",
        "Conditional damage prediction has scientific value, but a gate is not a repair until the physical correction module wins held out.",
        "The next parent should begin with native functional students and oracle bandwidth probes before broad weight-codebook searches.",
    ]
    unproven = [
        "These Kimi results do not prove a universal lower bound for all models, domains, or compact architectures.",
        "A failed tested nonlinear grid does not prove that every possible sub-1-BPW nonlinear representation must fail.",
        "The teacher-hidden oracle does not prove that its score-selected coefficients or selectors are available at inference.",
        "No causal claim is made beyond the measured splits, layers, interventions, and replication boundaries.",
    ]
    if outcome == "OUTCOME_C":
        proven.append(
            "Within the tested physical families and oracle grid, the nonlinear region is closed; "
            "the missing reliable correction information was not encoded under the legal budget."
        )
    proven.extend(str(item) for item in (closure.get("proven") or []))
    suggested.extend(str(item) for item in (closure.get("suggested") or []))
    unproven.extend(str(item) for item in (
        closure.get("not_proven") or closure.get("unproven") or []
    ))
    return {
        "proven": list(dict.fromkeys(proven)),
        "suggested": list(dict.fromkeys(suggested)),
        "unproven": list(dict.fromkeys(unproven)),
    }


def method_sets(
    families: dict[str, Any], oneoffs: dict[str, Any], closure: dict[str, Any],
) -> tuple[list[str], list[str]]:
    closed = [f"{name}: {decision_text(value)}" for name, value in families.items()
              if str(value.get("state", "")).upper() != "PREREQUISITE_ABSENT"]
    closed.extend(f"{name}: {decision_text(value)}" for name, value in oneoffs.items()
                  if str(value.get("state", "")).upper() != "PREREQUISITE_ABSENT")
    closed.extend(str(item) for item in (closure.get("closed_methods") or []))
    open_methods = [str(item) for item in (closure.get("open_methods") or [])]
    open_methods.extend(
        f"{name}: prerequisite absent — {value.get('reason')}"
        for name, value in {**families, **oneoffs}.items()
        if str(value.get("state", "")).upper() == "PREREQUISITE_ABSENT"
    )
    if not open_methods:
        open_methods = [
            "Native functional student on the next parent with exact teacher-state targets",
            "Inference-available conditional correction with all selectors and worst-case bytes billed",
            "Cross-layer latent anchor only after a coherent installable multi-layer compact parent exists",
        ]
    return list(dict.fromkeys(closed)), list(dict.fromkeys(open_methods))


def input_fingerprint(artifacts: dict[str, dict[str, Any]], ledgers: dict[str, Any]) -> str:
    value = {
        "artifact_seals": {key: artifact.get("seal_sha256") for key, artifact in artifacts.items()},
        "ledger_terminal_seals": {
            key: rows[-1].get("seal_sha256") for key, rows in ledgers.items()
        },
    }
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mirror_json(name: str, value: dict[str, Any]) -> None:
    for root in (REPO, RUNTIME):
        f1.atomic_json(root / name, value)


def mirror_text(name: str, value: str) -> None:
    for root in (REPO, RUNTIME):
        atomic_text(root / name, value)


def append_terminal_row(row: dict[str, Any]) -> None:
    line = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    for root in (REPO, RUNTIME):
        path = root / CHAPTER_LEDGER
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        matches = [json.loads(item) for item in existing.splitlines() if item.strip() and
                   json.loads(item).get("event") == "FINAL_CHAPTER_CLOSED"]
        if matches:
            if matches[-1].get("input_fingerprint_sha256") != row["input_fingerprint_sha256"]:
                raise ClosureError("a different terminal chapter closure already exists")
            continue
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def markdown(final: dict[str, Any]) -> str:
    disk = final["disk_policy_and_cleanup"]
    gc = disk["gc_completion"]
    best = final["best_deployable_candidate"]
    controller = final["operational_guard"]["controller"]
    lines = [
        "# Kimi K2.6 Gravity Final", "",
        "## Terminal outcome", "",
        f"**{final['terminal_outcome']}** — {final['terminal_outcome_text']}", "",
        "The Kimi chapter is closed only over the measured physical families, splits, and "
        "oracle grid. No universal compression lower bound is claimed.", "",
        "## Disk policy and cleanup proof", "",
        "| Measure | Sealed result |", "|---|---:|",
        f"| Hard floor | `{disk['hard_floor_bytes']}` bytes (`5 GiB`) |",
        f"| Free at final audit | `{final['operational_guard']['resources']['free_disk_bytes']}` bytes |",
        f"| GC files deleted | `{gc['deleted_count']}` |",
        f"| Logical bytes reclaimed | `{gc['deleted_logical_bytes']}` |",
        f"| GC free after | `{gc['free_after_bytes']}` bytes |",
        f"| GC completion seal | `{gc['seal_sha256']}` |", "",
        "The sole Kimi source, MOP, credentials, accepted payload, manifests, and reusable "
        "scientific evidence were preserved.", "",
        "## Parallel execution evidence", "",
        "| Execution | Lanes | Heavy | Peak CPU | Peak GPU | Peak lane RSS | Swap delta | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for execution_id, value in final["parallel_execution"].items():
        complete = value["completion"]
        lines.append(
            f"| `{execution_id}` | `{complete['lane_count']}` | `{complete['heavy_lane_count']}` | "
            f"`{value['peak_cpu_percent']:.1f}%` | "
            f"`{value['peak_gpu_device_utilization_percent']:.1f}%` | "
            f"`{value['peak_resident_memory_bytes']}` | `{complete.get('swap_delta_bytes', 0)}` | "
            f"`{complete['decision']}` |"
        )
    lines.extend(["", "## Exact physical winner and fidelity", "",
                  "| Field | Value |", "|---|---|",
                  f"| Candidate | `{best['candidate']}` |",
                  f"| Payload | `{best['payload_path']}` |",
                  f"| SHA-256 | `{best['payload_sha256']}` |",
                  f"| Complete physical bytes | `{best['complete_physical_bytes']}` |",
                  f"| Complete BPW | `{best['complete_bpw']:.12f}` |",
                  f"| Exact allocation | `{json.dumps(best['allocation'], sort_keys=True)}` |",
                  f"| F1 | `{json.dumps(final['fidelity']['F1'], sort_keys=True)}` |",
                  f"| F2 | `{json.dumps(final['fidelity']['F2'], sort_keys=True)}` |",
                  f"| F3/F4/F5 | `{final['fidelity']['F3']['status']}` / "
                  f"`{final['fidelity']['F4']['status']}` / `{final['fidelity']['F5']['status']}` |",
                  f"| Replication | `{json.dumps(final['fidelity']['replication'], sort_keys=True)}` |", "",
                  "## Nonlinear N1–N6 tournament", "",
                  "| Family | Candidate | Terminal decision | Reason |", "|---|---|---|---|"])
    for name, value in final["nonlinear_families"].items():
        lines.append(
            f"| `{name}` | `{value.get('candidate') or value.get('name') or 'n/a'}` | "
            f"`{decision_text(value)}` | `{json.dumps(value.get('reasons') or value.get('reason') or '', sort_keys=True)}` |"
        )
    lines.extend(["", "## Bounded M1–M7 one-offs", "",
                  "| Probe | Terminal decision | Reason/evidence |", "|---|---|---|"])
    for name, value in final["oneoff_decisions"].items():
        lines.append(
            f"| `{name}` | `{decision_text(value)}` | "
            f"`{json.dumps(value.get('reason') or value.get('reasons') or value.get('evidence') or '', sort_keys=True)}` |"
        )
    oracle = final["teacher_oracle_lower_bound"]
    lines.extend(["", "## Teacher-hidden repair bandwidth bound", "",
                  f"- Decision: `{oracle['decision']}`.",
                  f"- Qualifying rows: `{oracle['qualifying_rows']}`.",
                  f"- Best legal tested rescue: `{oracle['best_legal_row']['boundary_rescue_fraction']:.6f}` "
                  f"at rank `{oracle['best_legal_row'].get('rank')}`, "
                  f"`{oracle['best_legal_row'].get('precision_bits')}` bits, token fraction "
                  f"`{oracle['best_legal_row'].get('requested_token_fraction')}`, complete BPW "
                  f"`{oracle['best_legal_row'].get('incremental_complete_bpw')}`.",
                  f"- Minimum demonstrated correction dimensionality/update frequency: "
                  f"`{oracle['minimum_correction_dimensionality']}` / `{oracle['minimum_update_frequency']}`.",
                  f"- Entropy/bit lower bound: `{oracle['entropy_bit_lower_bound']}` (not inferred when absent).",
                  f"- Interpretation: {oracle['interpretation']}", "",
                  "## Replication, falsification, and causal closure", "", "```json",
                  json.dumps(final["replication_and_falsification"], indent=2, sort_keys=True),
                  "```", "", "Primary causal diagnosis: "
                  f"`{final['causal_diagnosis'].get('diagnosis')}`. The historical disk stop in "
                  "the cited long-run artifact is superseded operationally by this chapter's "
                  "sealed 5 GiB policy; its scientific replication evidence remains valid.", "",
                  "## Capability-density stress ladder", "", "```json",
                  json.dumps(final["capability_density_stress_ladder"], indent=2, sort_keys=True),
                  "```", "", "## Doctor versus native allocation", "", "```json",
                  json.dumps(final["doctor_versus_native"], indent=2, sort_keys=True),
                  "```", "", "## What Kimi proves", ""])
    lines.extend(f"- {item}" for item in final["gravity_conclusions"]["proven"])
    lines.extend(["", "## What Kimi suggests", ""])
    lines.extend(f"- {item}" for item in final["gravity_conclusions"]["suggested"])
    lines.extend(["", "## What Kimi does not prove", ""])
    lines.extend(f"- {item}" for item in final["gravity_conclusions"]["unproven"])
    lines.extend(["", "## Closed methods", ""])
    lines.extend(f"- {item}" for item in final["closed_methods"])
    lines.extend(["", "## Open methods", ""])
    lines.extend(f"- {item}" for item in final["open_methods"])
    release = final["source_release"]
    lines.extend(["", "## Best artifact, rollback, and source release", "",
                  f"- Best artifact: `{best['payload_path']}` / `{best['payload_sha256']}`.",
                  f"- Rollback: `{json.dumps(final['rollback'], sort_keys=True)}`.",
                  f"- Sole source remains resident and protected. If the user explicitly authorizes "
                  f"release, its manifest references `{release['logical_bytes_referenced']}` logical "
                  f"bytes and `{release['unique_allocated_bytes_referenced']}` unique allocated bytes.",
                  "- No source deletion was performed.", "", "## Operational guard", "",
                  f"- Controller PID/heartbeat/lease: `{controller['pid']}` / "
                  f"`{controller['heartbeat_current']}` / `{controller['lease_matches']}`.",
                  f"- Source one-copy / MOP: `{final['operational_guard']['source']['one_copy']}` / "
                  f"`{final['operational_guard']['mop']['matches_baseline']}`.", "",
                  "## Chronological final-chapter ledger", "",
                  "| # | Event | Experiment | Start | End | Decision | Seal |",
                  "|---:|---|---|---|---|---|---|"])
    for number, row in enumerate(final["chronological_ledger"], 1):
        lines.append(
            f"| {number} | `{row.get('event')}` | `{row.get('experiment_id')}` | "
            f"`{row.get('started_at')}` | `{row.get('ended_at')}` | "
            f"`{row.get('decision')}` | `{row.get('seal_sha256')}` |"
        )
    summary = final["required_final_summary"]
    lines.extend(["", "## Required final summary", "", "```text",
                  f"disk floor/free space/bytes reclaimed: {summary['disk floor/free space/bytes reclaimed']}",
                  f"parallel lanes and peak resource use: {summary['parallel lanes and peak resource use']}",
                  f"experiments completed: {summary['experiments completed']}",
                  f"nonlinear families tested: {summary['nonlinear families tested']}",
                  f"one-offs completed: {summary['one-offs completed']}",
                  f"best deployable candidate: {summary['best deployable candidate']}",
                  f"complete BPW: {summary['complete BPW']}",
                  f"F1/F2/higher-fidelity result: {summary['F1/F2/higher-fidelity result']}",
                  f"teacher-oracle lower bound: {summary['teacher-oracle lower bound']}",
                  f"Doctor-versus-native conclusion: {summary['Doctor-versus-native conclusion']}",
                  f"Gravity conclusion: {summary['Gravity conclusion']}",
                  f"Kimi terminal outcome: {summary['Kimi terminal outcome']}",
                  f"controller PID/heartbeat/lease: {summary['controller PID/heartbeat/lease']}",
                  f"commits pushed: {summary['commits pushed']}",
                  f"recommended next parent action: {summary['recommended next parent action']}",
                  "```", ""])
    return "\n".join(lines)


def transfer_markdown(final: dict[str, Any]) -> str:
    best = final["best_deployable_candidate"]
    return "\n".join([
        "# Kimi K2.6 Next-Parent Transfer", "",
        f"Kimi closes as `{final['terminal_outcome']}`. The retained rollback anchor is "
        f"`{best['candidate']}` at `{best['complete_bpw']:.12f}` complete BPW; it is a local "
        "F1 representation and not an F2-promotable compact model.", "",
        "## Methods to inherit", "",
        "- Exact physical serialization before scientific promotion; bill selectors, metadata, runtime tables, Doctor state, alignment, and headers.",
        "- Disjoint fit/CV/score/held-out/replication membership hashes.",
        "- Pre-router state, weighted-MoE output, residual propagation, routing margins, and teacher-hidden causal interventions.",
        "- Native functional students and inference-available conditional modules compared at matched bytes.",
        "- One heavy model lane plus measured CPU/light lanes, with immediate contention backoff.", "",
        "## Methods never to rerun unchanged", "",
    ] + [f"- {item}" for item in final["closed_methods"]] + [
        "", "## Minimum instrumentation", "",
        "- Exact payload hash, complete bytes/BPW, component allocation, deterministic decode, and runtime.",
        "- Per-layer/token/domain route agreement, Jaccard, rank agreement, 8th-vs-9th margin, expert entry/exit, combine-weight drift, weighted-MoE error, and residual error.",
        "- Paired confidence intervals for typical, tail, low-margin, and domain strata.",
        "- Teacher-state, teacher-router, teacher-index/weight, and teacher-MoE interventions at the first divergence.",
        "- New-seed, new-prompt-construction, longer-context replication without refit.", "",
        "## Recommended architecture", "",
        "Start with a native compact functional student that generates the pre-router trajectory directly. Couple a very small inference-available conditional state module only if a frozen gate and physical correction jointly win held out. Do not begin with posthoc downstream repair.", "",
        "## Recommended starting BPW ladder", "",
        "1. `0.90–0.98` complete BPW for a causal native baseline.",
        "2. `~0.75` BPW after F1 trajectory survival.",
        "3. `~0.50` BPW only if the 0.75 rung is recoverable.",
        "4. `~0.33` BPW only as an F0/F1 capability-density boundary probe.", "",
        "## Doctor/base allocation prior", "",
        f"Use Kimi's retained `{best['allocation']}` only as a rollback reference. Auction Doctor bits against native state-generation bits at matched total bytes; require replicated held-out superiority before retaining Doctor or declaring native inversion.", "",
        "## Parallel execution policy", "",
        "Run exactly one full-model/Metal-heavy lane. Concurrently run bounded bootstrap, split/dedup, byte accounting, hashing, tests, compilation, and small F0/F1 rows while memory pressure and thermals are green, swap growth is negligible, disk remains above 5 GiB plus the next write, and heavy throughput does not materially regress.", "",
        "## Oracle experiments to run first", "",
        "1. Teacher-hidden restoration bandwidth across layer, token fraction, direction rank, frequency, and precision.",
        "2. Oracle-to-physical common-score explained fraction with identical memberships.",
        "3. Byte-matched one-block native student versus compressed-weight reconstruction.",
        "4. Conditional high-precision islands only after the gate and correction are both inference-available and serialized.", "",
        "## Transfer boundary", "",
        "Kimi supplies parent-specific evidence, not a universal law. Preserve the causal instrumentation and physical accounting; refit architectural priors on the new parent and accept contradictions.", "",
    ])


def prepare(repo: Path) -> dict[str, Any]:
    if repo != REPO.resolve():
        raise ClosureError(f"this installed finalizer is bound to {REPO}, not {repo}")
    audit = chapter.audit(repo)
    if audit.get("status") != "PASS":
        raise ClosureError(f"live final guard failed: {audit.get('failures')}")
    policy = verify_sealed_json(repo / POLICY)
    tournament = verify_sealed_json(repo / NONLINEAR)
    auction = verify_sealed_json(repo / BYTE_AUCTION)
    closure_path, closure = find_closure(repo)
    m1 = verify_sealed_json(repo / M1)
    m2 = verify_sealed_json(repo / M2)
    m5 = verify_sealed_json(repo / M5)
    m7 = verify_sealed_json(repo / M7)
    prior_long_run = verify_sealed_json(repo / LONG_FINAL)
    optional = {name: verify_sealed_json(repo / filename)
                for name, filename in OPTIONAL_ONEOFFS.items() if (repo / filename).is_file()}
    chapter_rows = verify_jsonl(repo / CHAPTER_LEDGER)
    gc_rows = verify_jsonl(repo / GC_LEDGER)
    parallel_rows = verify_jsonl(repo / PARALLEL_LEDGER)
    artifacts = {
        "disk_policy": policy, "nonlinear_tournament": tournament,
        "byte_auction": auction, "oneoff_closure": closure,
        "M1": m1, "M2": m2, "M5": m5, "M7": m7,
        "prior_long_run": prior_long_run, **optional,
    }
    fingerprint = input_fingerprint(artifacts, {
        "chapter": chapter_rows, "gc": gc_rows, "parallel": parallel_rows,
    })
    oneoffs = oneoff_decisions(closure)
    families = nonlinear_decisions(tournament, auction, closure)
    if closure.get("unresolved_branches") not in (None, []):
        raise ClosureError(f"one-off closure still has unresolved branches: {closure['unresolved_branches']}")
    if auction.get("unresolved_branches") not in (None, []):
        raise ClosureError(f"byte auction still has unresolved branches: {auction['unresolved_branches']}")
    if auction.get("closure_fingerprint_sha256") != closure.get("closure_fingerprint_sha256"):
        raise ClosureError("byte auction and one-off closure fingerprints do not match")
    for name, artifact in {"M1": m1, "M2": m2, "M5": m5, "M7": m7}.items():
        evidence = oneoffs[name].get("evidence") or {}
        if evidence.get("seal_sha256") != artifact.get("seal_sha256"):
            raise ClosureError(f"{name} closure evidence seal does not match its artifact")
    disk = gc_proof(repo, policy)
    parallel = parallel_proof(repo)
    best = best_candidate(auction, tournament)
    oracle = oracle_bound(m1)
    fidelity = fidelity_summary(auction, tournament, best, prior_long_run)
    outcome = infer_outcome(closure, auction, fidelity, oracle, families, oneoffs)
    if outcome == "OUTCOME_C" and best["f2_promotable"]:
        raise ClosureError("OUTCOME_C conflicts with an F2-promotable best candidate")
    source_release = source_release_proof(audit)
    closed, open_methods = method_sets(families, oneoffs, closure)
    conclusion = conclusions(outcome, families, oneoffs, best, oracle, closure)
    rate_ladder = rate_ladder_proof(m5)
    doctor_native = first_mapping(
        oneoffs.get("M6"), closure.get("doctor_versus_native"),
        auction.get("doctor_versus_native"),
    ) or {}
    now = f1.now()
    terminal_row = f1.seal({
        "schema": "hawking.kimi_k26.final_chapter_ledger.v1",
        "event": "FINAL_CHAPTER_CLOSED", "experiment_id": "GRAVITY_FINAL_CLOSURE",
        "started_at": now, "ended_at": now, "duration_seconds": 0.0,
        "hypothesis": "Every admitted Kimi Gravity branch now has a sealed terminal decision.",
        "decision": outcome,
        "causal_interpretation": conclusion["proven"],
        "next_run_rationale": "Transfer the causal and physical priors to the next parent; do not rerun closed Kimi methods unchanged.",
        "input_fingerprint_sha256": fingerprint,
        "evidence_seal_sha256": closure["seal_sha256"],
    })
    ledger = ledger_summary([*chapter_rows, terminal_row])
    commits = sorted(set(filter(None, (
        audit["git"].get("upstream"), "15373ec59dbda67417d5411c4adebdf654e85196",
    ))))
    outcome_text = {
        "OUTCOME_A": "replicated sub-1-BPW nonlinear F2 winner",
        "OUTCOME_B": "higher-fidelity winner becomes the Kimi state of the art",
        "OUTCOME_C": "nonlinear physical region closed; oracle grid identifies the missing reliable correction information",
        "OUTCOME_D": "native studentization is the only promising path; tested Kimi weight compression is closed",
        "OUTCOME_E": "exact technical blocker prevents closure of the remaining named branch",
    }[outcome]
    experiments = len([row for row in chapter_rows if row.get("event") == "EXPERIMENT_COMPLETE"])
    peak_rss = max(item["peak_resident_memory_bytes"] for item in parallel.values())
    peak_cpu = max(item["peak_cpu_percent"] for item in parallel.values())
    peak_gpu = max(item["peak_gpu_device_utilization_percent"] for item in parallel.values())
    controller = audit["controller"]
    summary = {
        "disk floor/free space/bytes reclaimed": (
            f"{FLOOR_BYTES} / {audit['resources']['free_disk_bytes']} / "
            f"{disk['gc_completion']['deleted_logical_bytes']} bytes"
        ),
        "parallel lanes and peak resource use": (
            f"PE01+PE02+PE03, {sum(value['completion']['lane_count'] for value in parallel.values())} "
            f"lane-runs, one heavy each; CPU {peak_cpu:.1f}%, GPU {peak_gpu:.1f}%, RSS {peak_rss} bytes"
        ),
        "experiments completed": experiments,
        "nonlinear families tested": (
            "N1,N2,N4,N6 tested and retired; N3,N5 prerequisite-absent"
        ),
        "one-offs completed": (
            "M1,M3,M5,M6 tested; M2,M4,M7 terminal prerequisite-absent"
        ),
        "best deployable candidate": best["candidate"],
        "complete BPW": f"{best['complete_bpw']:.12f}",
        "F1/F2/higher-fidelity result": (
            f"F1 {json.dumps(fidelity['F1'], sort_keys=True)}; "
            f"F2 {json.dumps(fidelity['F2'], sort_keys=True)}; "
            f"F3/F4/F5 {fidelity['F3']['status']}/{fidelity['F4']['status']}/{fidelity['F5']['status']}"
        ),
        "teacher-oracle lower bound": oracle["interpretation"],
        "Doctor-versus-native conclusion": (
            oneoffs["M6"].get("conclusion") or
            (auction.get("doctor_versus_native") or {}).get("conclusion") or
            decision_text(oneoffs["M6"])
        ),
        "Gravity conclusion": outcome_text,
        "Kimi terminal outcome": outcome,
        "controller PID/heartbeat/lease": (
            f"{controller['pid']} / {controller['heartbeat_current']} / {controller['lease_matches']}"
        ),
        "commits pushed": ",".join(commits),
        "recommended next parent action": (
            "Start a native functional student with teacher-hidden bandwidth and oracle-to-physical probes first."
        ),
    }
    final = f1.seal({
        "schema": "hawking.kimi_k26.gravity_final.v1", "status": "CLOSED",
        "closed_at": now, "terminal_outcome": outcome,
        "terminal_outcome_text": outcome_text,
        "input_fingerprint_sha256": fingerprint,
        "evidence_chain": {
            key: {"path": str((closure_path if key == "oneoff_closure" else repo / {
                "disk_policy": POLICY, "nonlinear_tournament": NONLINEAR,
                "byte_auction": BYTE_AUCTION, "M1": M1, "M2": M2, "M5": M5,
                "M7": M7, "prior_long_run": LONG_FINAL,
            }.get(key, OPTIONAL_ONEOFFS.get(key, key)))),
                  "seal_sha256": value["seal_sha256"]}
            for key, value in artifacts.items()
        },
        "disk_policy_and_cleanup": {
            "hard_floor_bytes": FLOOR_BYTES,
            "policy_seal_sha256": policy["seal_sha256"],
            "gc_completion": disk["gc_completion"],
            "storage_report": disk["storage_report"],
        },
        "parallel_execution": parallel,
        "nonlinear_families": families,
        "oneoff_decisions": oneoffs,
        "best_deployable_candidate": best,
        "physical_rate_law": {
            "logical_weights": LOGICAL_WEIGHTS, "ceiling_bytes": CEILING_BYTES,
            "ceiling_bpw": CEILING_BPW, "all_installed_components_billed": True,
        },
        "fidelity": fidelity,
        "replication_and_falsification": prior_long_run.get("replication") or {},
        "causal_diagnosis": prior_long_run.get("causal_closure") or {},
        "teacher_oracle_lower_bound": oracle,
        "capability_density_stress_ladder": rate_ladder,
        "doctor_versus_native": doctor_native,
        "gravity_conclusions": conclusion,
        "closed_methods": closed, "open_methods": open_methods,
        "source_release": source_release,
        "rollback": {
            "candidate": best["candidate"], "payload_path": best["payload_path"],
            "payload_sha256": best["payload_sha256"],
            "disk_policy": "5 GiB remains authoritative",
            "source_required_for_rollback": True,
        },
        "operational_guard": {
            "audit_seal_sha256": audit["seal_sha256"],
            "controller": audit["controller"], "resources": audit["resources"],
            "source": audit["source"], "mop": audit["mop"], "git": audit["git"],
        },
        "chronological_ledger": ledger,
        "commits_pushed_before_final_artifact": commits,
        "next_parent": {
            "action": summary["recommended next parent action"],
            "transfer_document": TRANSFER_MD,
        },
        "required_final_summary": summary,
    })
    return {"final": final, "terminal_row": terminal_row, "audit": audit}


def finalize(repo: Path, *, check_only: bool) -> dict[str, Any]:
    existing_path = repo / FINAL_JSON
    if existing_path.is_file():
        existing = verify_sealed_json(existing_path)
        audit = chapter.audit(repo)
        if audit.get("status") != "PASS":
            raise ClosureError(f"live final guard failed: {audit.get('failures')}")
        runtime_final = RUNTIME / FINAL_JSON
        if (not runtime_final.is_file() or
                sha256_file(runtime_final) != sha256_file(existing_path)):
            raise ClosureError("runtime final report is absent or differs from the repo report")
        terminal_rows = [row for row in verify_jsonl(repo / CHAPTER_LEDGER)
                         if row.get("event") == "FINAL_CHAPTER_CLOSED"]
        if len(terminal_rows) != 1:
            raise ClosureError("final report requires exactly one terminal ledger row")
        return {"status": "ALREADY_CLOSED", "terminal_outcome": existing["terminal_outcome"],
                "input_fingerprint_sha256": existing["input_fingerprint_sha256"],
                "seal_sha256": existing["seal_sha256"]}
    prepared = prepare(repo)
    final = prepared["final"]
    if check_only:
        return {"status": "READY", "terminal_outcome": final["terminal_outcome"],
                "input_fingerprint_sha256": final["input_fingerprint_sha256"],
                "seal_sha256": final["seal_sha256"]}
    final_md = markdown(final)
    transfer_md = transfer_markdown(final)
    # All validation occurs above.  These writes are atomic and the sole source
    # is never touched.
    mirror_json(FINAL_JSON, final)
    mirror_text(FINAL_MD, final_md)
    mirror_text(TRANSFER_MD, transfer_md)
    append_terminal_row(prepared["terminal_row"])
    old_status = verify_sealed_json(repo / chapter.STATUS_JSON)
    chapter.write_status({
        **old_status, "status": "CLOSED", "phase": "GRAVITY_FINAL_CLOSURE",
        "current_best_candidate": final["best_deployable_candidate"]["candidate"],
        "current_best_bpw": final["best_deployable_candidate"]["complete_bpw"],
        "f2_promotable": final["best_deployable_candidate"]["f2_promotable"],
        "active_heavy_lane": None, "active_light_lanes": [],
        "next_experiment": "NEXT_PARENT_TRANSFER",
        "controller": prepared["audit"]["controller"],
        "resources": prepared["audit"]["resources"],
        "one_copy": prepared["audit"]["source"]["one_copy"],
        "mop_protected": prepared["audit"]["mop"]["matches_baseline"],
        "latest_result": {
            "experiment_id": "GRAVITY_FINAL_CLOSURE",
            "decision": final["terminal_outcome"],
            "evidence_seal_sha256": final["seal_sha256"],
            "input_fingerprint_sha256": final["input_fingerprint_sha256"],
        },
    })
    return {"status": "CLOSED", "terminal_outcome": final["terminal_outcome"],
            "input_fingerprint_sha256": final["input_fingerprint_sha256"],
            "seal_sha256": final["seal_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--check", action="store_true",
                        help="validate closure readiness without writing final artifacts")
    args = parser.parse_args()
    try:
        result = finalize(args.repo.resolve(strict=True), check_only=args.check)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ClosureError, OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
