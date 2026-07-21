#!/usr/bin/env python3.12
"""Seal the terminal Kimi Gravity byte auction and one-off adjudication.

The closure is intentionally evidence-only.  It never trains a candidate or
runs a model forward.  A branch is classified as one of:

* TESTED_PROMOTED / TESTED_REJECTED / TESTED_COMPLETE;
* PREREQUISITE_ABSENT (the named experiment is scientifically inadmissible);
* UNRESOLVED_JUSTIFIED_BRANCH (closure must not be claimed).

Missing N3/N5 or physical one-off derivatives become terminal prerequisite
rejections only when the sealed contextual tournament has no F1 survivor.
That rule prevents a missing artifact from being mislabeled as a failed test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


LOGICAL_WEIGHTS = 44_040_192
COMPLETE_CEILING_BYTES = 5_394_923
CURRENT_PARENT_BYTES = 5_001_815
CURRENT_PARENT_BPW = CURRENT_PARENT_BYTES * 8 / LOGICAL_WEIGHTS
RUNTIME_DEFAULT = Path.home() / "Library/Application Support/Hawking/KimiK26"

TOURNAMENT = "KIMI_K26_GRAVITY_NONLINEAR_TOURNAMENT.json"
CONTEXTUAL_SEAM = "KIMI_K26_GRAVITY_CONTEXTUAL_SEAM.json"
M1 = "KIMI_K26_GRAVITY_M1_ORACLE_BANDWIDTH.json"
M2 = "KIMI_K26_GRAVITY_M2_CONDITIONAL_GATE.json"
M5 = "KIMI_K26_GRAVITY_RATE_LADDER.json"
M7 = "KIMI_K26_GRAVITY_M7_ORACLE_GAP.json"
HOOKS = "KIMI_K26_GRAVITY_ONEOFF_HOOKS.json"
BYTE_AUCTION = "KIMI_K26_FINAL_BYTE_AUCTION.json"
CLOSURE = "KIMI_K26_GRAVITY_ONEOFF_CLOSURE.json"
STATUS_JSON = "KIMI_K26_FINAL_CHAPTER_STATUS.json"
STATUS_MD = "KIMI_K26_FINAL_CHAPTER_STATUS.md"
LEDGER = "KIMI_K26_FINAL_CHAPTER_LEDGER.jsonl"

OPTIONAL_ARTIFACTS = {
    "N3": (
        "KIMI_K26_GRAVITY_N3_ASYMMETRIC_ALLOCATION.json",
        "KIMI_K26_GRAVITY_N3.json",
    ),
    "N5": (
        "KIMI_K26_GRAVITY_N5_COLD_EXPERTS.json",
        "KIMI_K26_GRAVITY_N5.json",
    ),
    "M1_DOWNSTREAM": (
        "KIMI_K26_GRAVITY_M1_DOWNSTREAM.json",
        "KIMI_K26_GRAVITY_M1_F2.json",
    ),
    "M2_PHYSICAL": (
        "KIMI_K26_GRAVITY_M2_PHYSICAL.json",
        "KIMI_K26_GRAVITY_M2_F2.json",
    ),
    "M3": ("KIMI_K26_GRAVITY_M3_STUDENTIZATION.json",),
    "M4": ("KIMI_K26_GRAVITY_M4_CROSS_LAYER_ANCHOR.json",),
    "M6": ("KIMI_K26_GRAVITY_M6_DOCTOR_INVERSION.json",),
}


class ClosureError(RuntimeError):
    """Fail-closed closure or evidence-validation error."""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    return {**unsigned, "seal_sha256": hashlib.sha256(canonical(unsigned)).hexdigest()}


def verify_seal(value: dict[str, Any], label: str) -> str:
    expected = value.get("seal_sha256")
    if not isinstance(expected, str):
        raise ClosureError(f"{label} has no evidence seal")
    actual = seal(value)["seal_sha256"]
    if actual != expected:
        raise ClosureError(f"{label} seal mismatch: {actual} != {expected}")
    return actual


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"required JSON absent or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ClosureError(f"JSON root is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_receipt(path: Path, expected: str | None, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ClosureError(f"{label} payload missing: {resolved}")
    actual = sha256_file(resolved)
    if expected is not None and actual != expected:
        raise ClosureError(f"{label} payload hash mismatch: {actual} != {expected}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_sealed(path: Path, *, required: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not path.is_file():
        if required:
            raise ClosureError(f"required sealed artifact absent: {path}")
        return None, {"status": "ABSENT", "path": str(path)}
    value = read_json(path)
    evidence_seal = verify_seal(value, path.name)
    return value, {
        "status": "PRESENT_AND_SEALED", "path": str(path.resolve()),
        "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "seal_sha256": evidence_seal,
    }


def discover_optional(repo: Path, names: Iterable[str]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    searched = []
    for name in names:
        path = repo / name
        searched.append(str(path))
        if path.is_file():
            return load_sealed(path)
    return None, {"status": "ABSENT", "searched": searched}


def positive_decision(decision: Any) -> bool:
    if not isinstance(decision, str):
        return False
    upper = decision.upper()
    return any(word in upper for word in ("PROMOTE", "SURVIVE", "WINNER", "ADVANCE")) and not any(
        word in upper for word in ("RETIRE", "REJECT", "FAIL", "NO_")
    )


def negative_decision(decision: Any) -> bool:
    if not isinstance(decision, str):
        return False
    upper = decision.upper()
    return any(word in upper for word in ("RETIRE", "REJECT", "FAIL", "NO_"))


def tested_adjudication(
    identifier: str, decision: Any, evidence: dict[str, Any], *,
    tested_scope: str, fallback_unresolved: str,
) -> dict[str, Any]:
    if positive_decision(decision):
        state = "TESTED_PROMOTED"
    elif negative_decision(decision):
        state = "TESTED_REJECTED"
    else:
        state = "UNRESOLVED_JUSTIFIED_BRANCH"
    return {
        "id": identifier, "state": state, "source_decision": decision,
        "tested_scope": tested_scope, "evidence": evidence,
        "reason": fallback_unresolved if state == "UNRESOLVED_JUSTIFIED_BRANCH" else None,
    }


def verify_payload_object(payload: Any, label: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
        return None
    receipt = file_receipt(Path(payload["path"]), payload.get("sha256"), label)
    declared = payload.get("bytes")
    if isinstance(declared, int) and receipt["bytes"] != declared:
        raise ClosureError(
            f"{label} payload size mismatch: {receipt['bytes']} != {declared}"
        )
    return {**payload, **receipt, "hash_verified": True}


def verify_embedded_receipts(
    value: Any, label: str, cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify every embedded file receipt with a path and SHA-256 exactly once."""
    verified: list[dict[str, Any]] = []
    if isinstance(value, dict):
        path_value = value.get("path")
        digest = value.get("sha256")
        if (isinstance(path_value, str) and isinstance(digest, str) and
                re.fullmatch(r"[0-9a-f]{64}", digest)):
            resolved = str(Path(path_value).expanduser().resolve())
            if resolved in cache:
                if cache[resolved]["sha256"] != digest:
                    raise ClosureError(f"{label} gives conflicting hashes for {resolved}")
                receipt = cache[resolved]
            else:
                receipt = file_receipt(Path(resolved), digest, label)
                cache[resolved] = receipt
            declared = value.get("bytes")
            if isinstance(declared, int) and declared != receipt["bytes"]:
                raise ClosureError(f"{label} embedded receipt size mismatch for {resolved}")
            verified.append(receipt)
        for key, item in value.items():
            verified.extend(verify_embedded_receipts(item, f"{label}.{key}", cache))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            verified.extend(verify_embedded_receipts(item, f"{label}[{index}]", cache))
    return verified


def verify_tournament_rows(tournament: dict[str, Any], runtime: Path) -> list[dict[str, Any]]:
    rows = tournament.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ClosureError("nonlinear tournament has no rows")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ClosureError(f"tournament row {index} is not an object")
        row_seal = row.get("seal_sha256")
        if not isinstance(row_seal, str):
            raise ClosureError(f"tournament row {index} has no source-result seal")
        source_result_path = (
            runtime / "final_chapter/gravity_nonlinear" /
            f"{row.get('candidate')}_RESULT.json"
        )
        source_result, _ = load_sealed(source_result_path)
        assert source_result is not None
        if source_result.get("seal_sha256") != row_seal:
            raise ClosureError(f"tournament row {index} source-result seal mismatch")
        payload = verify_payload_object(
            row.get("physical_payload"), f"tournament row {index} {row.get('candidate')}",
        )
        if payload is None:
            raise ClosureError(f"tournament row {index} lacks a physical payload")
        complete_bytes = payload.get("bytes")
        if not isinstance(complete_bytes, int):
            raise ClosureError(f"tournament row {index} lacks complete payload bytes")
        calculated_bpw = complete_bytes * 8 / LOGICAL_WEIGHTS
        declared_bpw = payload.get("complete_bpw")
        if not isinstance(declared_bpw, (float, int)) or abs(calculated_bpw - declared_bpw) > 1e-12:
            raise ClosureError(f"tournament row {index} BPW arithmetic mismatch")
        result.append({
            "family": row.get("family"), "candidate": row.get("candidate"),
            "ablation_of": row.get("ablation_of"), "row_seal_sha256": row_seal,
            "physical_payload": payload, "complete_physical_bytes": complete_bytes,
            "complete_bpw": calculated_bpw,
            "within_0_98_bpw": complete_bytes <= COMPLETE_CEILING_BYTES,
            "deterministic_decode": row.get("f0", {}).get("deterministic_decode"),
            "teacher_access_at_inference": row.get("f0", {}).get("teacher_access_at_inference"),
            "frozen_score_metrics": row.get("frozen_score_metrics"),
        })
    return result


def family_decisions(tournament: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    raw = tournament.get("family_decisions")
    if not isinstance(raw, dict):
        raise ClosureError("tournament has no family_decisions")
    decisions = {}
    survivors = []
    for family in ("N1", "N2", "N4", "N6"):
        value = raw.get(family)
        if not isinstance(value, dict):
            raise ClosureError(f"tournament lacks {family} decision")
        decision = value.get("decision")
        adjudication = tested_adjudication(
            family, decision,
            {"tournament_seal_sha256": tournament["seal_sha256"], "family_record": value},
            tested_scope="CONTEXTUAL_LAYER1_EXPERT0_F0_F1",
            fallback_unresolved="Tournament family decision is neither promotion nor rejection.",
        )
        decisions[family] = adjudication
        if adjudication["state"] == "TESTED_PROMOTED":
            survivors.append(family)
    return decisions, survivors


def optional_family(
    family: str, artifact: dict[str, Any] | None, receipt: dict[str, Any],
    survivors: list[str], prerequisite: str,
) -> dict[str, Any]:
    if artifact is not None:
        decision = artifact.get("decision")
        return tested_adjudication(
            family, decision, receipt, tested_scope=artifact.get("claim_boundary", "DECLARED_ARTIFACT_SCOPE"),
            fallback_unresolved=f"{family} artifact has no terminal decision.",
        )
    if not survivors:
        return {
            "id": family, "state": "PREREQUISITE_ABSENT",
            "tested_scope": None, "evidence": receipt,
            "reason": prerequisite,
            "scientific_interpretation": (
                "This is not a tested failure. The experiment is inadmissible because its "
                "required parent representation did not survive contextual F1."
            ),
        }
    return {
        "id": family, "state": "UNRESOLVED_JUSTIFIED_BRANCH",
        "tested_scope": None, "evidence": receipt,
        "reason": f"A contextual F1 survivor exists, so missing {family} evidence cannot be closed.",
    }


def adjudicate_m1(
    artifact: dict[str, Any] | None, receipt: dict[str, Any],
    downstream: dict[str, Any] | None, downstream_receipt: dict[str, Any],
) -> dict[str, Any]:
    if artifact is None:
        return {"id": "M1", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
                "reason": "M1 cached oracle bandwidth is independently runnable and absent."}
    if artifact.get("qualifying_rows") == 0 and artifact.get("decision") == "NO_CACHED_BOUNDARY_ROW_JUSTIFIES_REAL_FORWARD":
        return {
            "id": "M1", "state": "TESTED_REJECTED", "tested_scope": artifact.get("claim_boundary"),
            "source_decision": artifact.get("decision"), "evidence": receipt,
            "reason": "No cached boundary row met the preregistered forward-admission CI.",
            "deployable": False, "oracle_only": True,
        }
    if downstream is not None:
        return tested_adjudication(
            "M1", downstream.get("decision"), downstream_receipt,
            tested_scope=downstream.get("claim_boundary", "CACHED_STATE_DOWNSTREAM_FORWARD"),
            fallback_unresolved="M1 downstream artifact has no terminal decision.",
        )
    return {
        "id": "M1", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
        "reason": "At least one M1 row qualified, but the required downstream forward is absent.",
    }


def adjudicate_m2(
    artifact: dict[str, Any] | None, receipt: dict[str, Any],
    physical: dict[str, Any] | None, physical_receipt: dict[str, Any],
    survivors: list[str],
) -> dict[str, Any]:
    if physical is not None:
        return tested_adjudication(
            "M2", physical.get("decision"), physical_receipt,
            tested_scope=physical.get("claim_boundary", "PHYSICAL_CONDITIONAL_ISLAND"),
            fallback_unresolved="M2 physical artifact has no terminal decision.",
        )
    if artifact is not None and negative_decision(artifact.get("decision")):
        return tested_adjudication(
            "M2", artifact.get("decision"), receipt,
            tested_scope="CACHED_CONDITIONAL_GATE", fallback_unresolved="",
        )
    if not survivors:
        return {
            "id": "M2", "state": "PREREQUISITE_ABSENT", "tested_scope": "GATE_DIAGNOSTIC_ONLY",
            "evidence": receipt,
            "reason": (
                "The gate may predict damage, but no physical nonlinear residual/high-precision "
                "module survived F1 to place behind it."
            ),
            "tested_gate_skill": artifact.get("skillful_trigger_rates") if artifact else None,
            "deployable": False,
        }
    return {
        "id": "M2", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
        "reason": "A nonlinear F1 survivor exists but its physical conditional-island test is absent.",
    }


def adjudicate_m3(
    artifact: dict[str, Any] | None, receipt: dict[str, Any],
    tournament: dict[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if artifact is not None:
        return tested_adjudication(
            "M3", artifact.get("decision"), receipt,
            tested_scope=artifact.get("claim_boundary", "NATIVE_STUDENTIZATION"),
            fallback_unresolved="M3 artifact has no terminal decision.",
        )
    n6 = tournament.get("family_decisions", {}).get("N6", {})
    native_rows = [row for row in rows if row["family"] == "N6" and row["ablation_of"] is None]
    if native_rows and negative_decision(n6.get("decision")):
        row = native_rows[0]
        return {
            "id": "M3", "state": "TESTED_REJECTED",
            "tested_scope": "N6_NATIVE_SWIGLU_DIRECT_FUNCTION_STUDENT_CONTEXTUAL_F1",
            "source_decision": n6.get("decision"),
            "evidence": {
                "tournament_seal_sha256": tournament["seal_sha256"],
                "row_seal_sha256": row["row_seal_sha256"],
                "candidate": row["candidate"], "payload": row["physical_payload"],
            },
            "reason": "The teacher-free native functional student was tested and retired at contextual F1.",
            "mapping_caveat": "M3 is closed only at the one-block direct-function scope exercised by N6.",
        }
    return {"id": "M3", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
            "reason": "No direct-function native student evidence is available."}


def adjudicate_m4(
    artifact: dict[str, Any] | None, receipt: dict[str, Any], survivors: list[str],
) -> dict[str, Any]:
    if artifact is not None:
        return tested_adjudication(
            "M4", artifact.get("decision"), receipt,
            tested_scope=artifact.get("claim_boundary", "CROSS_LAYER_STATE_ANCHOR"),
            fallback_unresolved="M4 artifact has no terminal decision.",
        )
    if not survivors:
        return {
            "id": "M4", "state": "PREREQUISITE_ABSENT", "evidence": receipt,
            "reason": "No coherent compact F1 survivor exists to install at early/middle/late sentinels.",
            "scientific_interpretation": "A synthetic multi-layer perturbation would not test M4.",
        }
    return {"id": "M4", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
            "reason": "A compact F1 survivor exists but cross-layer anchor evidence is absent."}


def adjudicate_m5(
    artifact: dict[str, Any] | None, receipt: dict[str, Any], survivors: list[str],
    verified_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if artifact is not None and artifact.get("status") == "PASS":
        promoted = [row for row in artifact.get("stress_ladder", [])
                    if row.get("strict_final_verdict") == "SURVIVES_STRICT_CONTEXTUAL_F1"]
        return {
            "id": "M5", "state": "UNRESOLVED_JUSTIFIED_BRANCH" if promoted else "TESTED_COMPLETE",
            "tested_scope": "F0_F1_RATE_STRESS_LADDER", "source_decision": artifact.get("decision"),
            "evidence": receipt, "rows": verified_rows,
            "reason": "A lower-rate row earned F2." if promoted else None,
        }
    if not survivors:
        return {
            "id": "M5", "state": "PREREQUISITE_ABSENT", "evidence": receipt,
            "reason": (
                "No nonlinear/native representation survived contextual F1 to define one coherent "
                "architecture for the 0.75/0.50/0.33 physical stress ladder."
            ),
            "scientific_interpretation": "This is not evidence that lower BPW is intrinsically impossible.",
        }
    return {"id": "M5", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
            "reason": "A survivor exists, so the missing or incomplete rate ladder remains justified."}


def verify_m5_rows(artifact: dict[str, Any] | None) -> list[dict[str, Any]]:
    if artifact is None:
        return []
    result = []
    for index, row in enumerate(artifact.get("stress_ladder", [])):
        if not isinstance(row, dict):
            raise ClosureError(f"M5 row {index} is not an object")
        payload = verify_payload_object(row.get("physical_payload"), f"M5 row {index}")
        if payload is None:
            raise ClosureError(f"M5 row {index} has no complete payload receipt")
        complete = payload.get("bytes")
        if not isinstance(complete, int) or payload["bytes"] != complete:
            raise ClosureError(f"M5 row {index} payload is not the complete candidate")
        calculated = complete * 8 / LOGICAL_WEIGHTS
        declared = row.get("physical_rate", {}).get("actual_complete_bpw")
        if not isinstance(declared, (int, float)) or abs(calculated - declared) > 1e-12:
            raise ClosureError(f"M5 row {index} BPW arithmetic mismatch")
        result.append({**row, "payload": payload, "actual_complete_bpw": calculated})
    return result


def adjudicate_m6(
    artifact: dict[str, Any] | None, receipt: dict[str, Any],
    tournament: dict[str, Any], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if artifact is not None:
        return tested_adjudication(
            "M6", artifact.get("decision"), receipt,
            tested_scope=artifact.get("claim_boundary", "DOCTOR_NATIVE_INVERSION"),
            fallback_unresolved="M6 artifact has no terminal decision.",
        )
    native = [row for row in rows if row["family"] == "N6" and row["ablation_of"] is None]
    n6 = tournament.get("family_decisions", {}).get("N6", {})
    if native and negative_decision(n6.get("decision")):
        row = native[0]
        allocation = row["physical_payload"]
        if allocation.get("doctor_component_bytes") != 0:
            raise ClosureError("N6 mapping to M6 is invalid: Doctor bytes are nonzero")
        return {
            "id": "M6", "state": "TESTED_REJECTED",
            "tested_scope": "FULL_NATIVE_ZERO_DOCTOR_CONTEXTUAL_F1",
            "source_decision": n6.get("decision"),
            "evidence": {
                "tournament_seal_sha256": tournament["seal_sha256"],
                "candidate": row["candidate"], "payload": allocation,
                "row_seal_sha256": row["row_seal_sha256"],
            },
            "reason": "The zero-Doctor native allocation was physically tested and failed contextual F1.",
            "conclusion": "Doctor remains justified locally; global optimality is not proven.",
        }
    return {"id": "M6", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
            "reason": "No zero-Doctor native allocation evidence is available."}


def adjudicate_m7(
    artifact: dict[str, Any] | None, receipt: dict[str, Any], survivors: list[str],
) -> dict[str, Any]:
    if artifact is None:
        return {"id": "M7", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
                "reason": "The cached baseline/linear/oracle common-score comparison is absent."}
    if artifact.get("status") == "PASS" and artifact.get("decision") == "M7_COMMON_SCORE_GAP_COMPLETE":
        return {"id": "M7", "state": "TESTED_COMPLETE", "evidence": receipt,
                "tested_scope": "COMMON_SCORE_ORACLE_TO_PHYSICAL_GAP",
                "variants": artifact.get("variants"), "source_decision": artifact.get("decision")}
    if not survivors and artifact.get("status") == "PARTIAL_WAITING_PREREQUISITE":
        return {
            "id": "M7", "state": "PREREQUISITE_ABSENT",
            "tested_scope": "BASELINE_LINEAR_TEACHER_ORACLE_COMMON_SCORE",
            "evidence": receipt, "tested_variants": artifact.get("variants"),
            "reason": "No nonlinear F1 survivor exists to supply BEST_NONLINEAR_PHYSICAL.",
            "scientific_interpretation": (
                "The baseline/linear/oracle fractions are tested; the nonlinear fraction is "
                "undefined rather than zero. Attribution remains MIXED_UNRESOLVED."
            ),
        }
    return {"id": "M7", "state": "UNRESOLVED_JUSTIFIED_BRANCH", "evidence": receipt,
            "reason": "A nonlinear survivor exists or the M7 artifact lacks a terminal common-score result."}


def parent_receipt(runtime: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = (
        runtime / "f1_representation_bracket/doctor_auction/"
        "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json"
    )
    result, result_receipt = load_sealed(result_path)
    assert result is not None
    payload = verify_payload_object(result.get("payload"), "current best P1")
    if payload is None:
        raise ClosureError("current best result has no physical payload")
    if payload["bytes"] != CURRENT_PARENT_BYTES:
        raise ClosureError("current best payload bytes differ from sealed baseline")
    return result, {"result": result_receipt, "payload": payload}


def validate_auxiliary_payloads(
    m1: dict[str, Any] | None, m2: dict[str, Any] | None,
) -> dict[str, Any]:
    oracle = []
    if m1 is not None:
        for index, row in enumerate(m1.get("downstream_forward_queue", [])):
            payload = verify_payload_object(
                row.get("serialized_oracle_payload"), f"M1 oracle payload {index}",
            )
            if payload is not None:
                oracle.append({
                    "layer": row.get("layer"), "rank": row.get("rank"),
                    "precision_bits": row.get("precision_bits"),
                    "token_fraction": row.get("actual_token_fraction"),
                    "boundary_rescue_ci95": row.get("boundary_rescue_ci95"),
                    "within_incremental_0_98_bpw": row.get("within_incremental_0_98_bpw"),
                    "payload": payload, "deployable": False,
                })
    gate = None
    if m2 is not None:
        gate = verify_payload_object(m2.get("serialized_gate_payload"), "M2 gate")
    return {"m1_oracle_payloads": oracle, "m2_gate_payload": gate}


def verify_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        raise ClosureError(f"ledger absent: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClosureError(f"ledger JSON invalid at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ClosureError(f"ledger record is not an object at {path}:{line_number}")
        verify_seal(record, f"{path.name}:{line_number}")
        records.append(record)
    return records


def append_mirrored_ledger(
    repo: Path, runtime: Path, record: dict[str, Any], fingerprint: str,
) -> dict[str, Any]:
    repo_path = repo / LEDGER
    runtime_path = runtime / LEDGER
    repo_records = verify_jsonl(repo_path)
    runtime_records = verify_jsonl(runtime_path)
    if canonical(repo_records) != canonical(runtime_records):
        raise ClosureError("repo/runtime final-chapter ledgers diverge")
    for existing in repo_records:
        if existing.get("event") == "GRAVITY_ONEOFF_CLOSURE" and existing.get(
            "closure_fingerprint_sha256"
        ) == fingerprint:
            return existing
    value = seal({
        "schema": "hawking.kimi_k26.final_chapter_ledger.v1", **record,
        "closure_fingerprint_sha256": fingerprint,
    })
    line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    for path in (repo_path, runtime_path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return value


def status_markdown(value: dict[str, Any]) -> str:
    resources = value.get("resources", {})
    controller = value.get("controller", {})
    return "\n".join((
        "# Kimi K2.6 Gravity Final Chapter Status", "",
        f"- Status: **{value.get('status')}**",
        f"- Phase: `{value.get('phase')}`",
        f"- Started: `{value.get('started_at')}`",
        f"- Updated: `{value.get('updated_at')}`",
        f"- Current best: `{value.get('current_best_candidate')}` / `{value.get('current_best_bpw')}` BPW",
        f"- F2 promotable: `{value.get('f2_promotable')}`",
        f"- Experiments completed: `{value.get('experiments_completed', 0)}`",
        f"- One-offs adjudicated: `{value.get('oneoffs_adjudicated', 0)}`",
        f"- Nonlinear families adjudicated: `{value.get('nonlinear_families_adjudicated', 0)}`",
        f"- Next experiment: `{value.get('next_experiment')}`", "",
        "## Guards", "",
        f"- Disk floor/free/headroom: `{resources.get('disk_floor_bytes', 0)}` / `{resources.get('free_disk_bytes', 0)}` / `{resources.get('disk_headroom_bytes', 0)}` bytes",
        f"- Controller PID/heartbeat/lease: `{controller.get('pid')}` / `{controller.get('heartbeat_current')}` / `{controller.get('lease_matches')}`",
        f"- Source one-copy / MOP: `{value.get('one_copy')}` / `{value.get('mop_protected')}`", "",
        "## Latest result", "", "```json",
        json.dumps(value.get("latest_result", {}), indent=2, sort_keys=True), "```", "",
    ))


def write_mirrored_status(
    repo: Path, runtime: Path, closure: dict[str, Any], ledger_record: dict[str, Any],
) -> dict[str, Any]:
    repo_status = read_json(repo / STATUS_JSON)
    runtime_status = read_json(runtime / STATUS_JSON)
    verify_seal(repo_status, STATUS_JSON)
    verify_seal(runtime_status, f"runtime/{STATUS_JSON}")
    if repo_status["seal_sha256"] != runtime_status["seal_sha256"]:
        raise ClosureError("repo/runtime final-chapter status diverges")
    unresolved = closure["unresolved_branches"]
    updated_at = now()
    started_at = str(repo_status.get("started_at", updated_at))
    wall_clock = (
        dt.datetime.fromisoformat(updated_at.replace("Z", "+00:00")) -
        dt.datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    ).total_seconds()
    updated = seal({
        **{key: value for key, value in repo_status.items() if key != "seal_sha256"},
        "status": "READY_FOR_FINAL_REPORT" if not unresolved else "MANAGING",
        "phase": "GRAVITY_ONEOFF_CLOSURE_COMPLETE" if not unresolved else "GRAVITY_CLOSURE_INCOMPLETE",
        "updated_at": updated_at, "wall_clock_seconds": wall_clock,
        "oneoffs_adjudicated": 7, "nonlinear_families_adjudicated": 6,
        "next_experiment": (
            "KIMI_GRAVITY_FINAL_AND_NEXT_PARENT_TRANSFER" if not unresolved else
            unresolved[0]
        ),
        "latest_result": {
            "experiment_id": "GRAVITY_N3_N5_M1_M7_TERMINAL_ADJUDICATION",
            "decision": closure["decision"],
            "terminal_outcome_recommendation": closure["terminal_outcome_recommendation"],
            "unresolved_branches": unresolved,
            "evidence_seal_sha256": closure["seal_sha256"],
            "ledger_seal_sha256": ledger_record["seal_sha256"],
        },
    })
    markdown = status_markdown(updated)
    for root in (repo, runtime):
        atomic_json(root / STATUS_JSON, updated)
        atomic_text(root / STATUS_MD, markdown)
    return updated


def mirror_artifact(repo: Path, runtime: Path, name: str, value: dict[str, Any]) -> None:
    for root in (repo, runtime):
        atomic_json(root / name, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--runtime", type=Path, default=RUNTIME_DEFAULT)
    args = parser.parse_args()
    try:
        repo = args.repo.expanduser().resolve(strict=True)
        runtime = args.runtime.expanduser().resolve(strict=True)
        tournament, tournament_receipt = load_sealed(repo / TOURNAMENT)
        seam, seam_receipt = load_sealed(repo / CONTEXTUAL_SEAM)
        m1, m1_receipt = load_sealed(repo / M1, required=False)
        m2, m2_receipt = load_sealed(repo / M2, required=False)
        m5, m5_receipt = load_sealed(repo / M5, required=False)
        m7, m7_receipt = load_sealed(repo / M7, required=False)
        hooks, hooks_receipt = load_sealed(repo / HOOKS, required=False)
        assert tournament is not None and seam is not None
        if tournament.get("contextual_capture_seal_sha256") != seam["seal_sha256"]:
            raise ClosureError("tournament does not bind the supplied contextual seam")
        rows = verify_tournament_rows(tournament, runtime)
        nonlinear, survivors = family_decisions(tournament)
        m5_rows = verify_m5_rows(m5)
        receipt_cache: dict[str, dict[str, Any]] = {}
        for label, artifact in (
            ("contextual_seam", seam), ("tournament", tournament),
            ("M1", m1), ("M2", m2), ("M5", m5), ("M7", m7),
            ("hooks", hooks),
        ):
            if artifact is not None:
                verify_embedded_receipts(artifact, label, receipt_cache)
        optional = {}
        optional_receipts = {}
        for key, names in OPTIONAL_ARTIFACTS.items():
            optional[key], optional_receipts[key] = discover_optional(repo, names)
            if optional[key] is not None:
                verify_embedded_receipts(optional[key], key, receipt_cache)
        prerequisites = {
            "N3": "N3 requires a surviving representation family plus multi-layer/expert sensitivity evidence to reallocate.",
            "N5": "N5 requires a viable contextual residual generator and cross-expert cold/high-use captures.",
        }
        explicit_family: dict[str, dict[str, Any]] = {}
        for family in ("N3", "N5"):
            embedded = tournament.get("family_decisions", {}).get(family)
            if optional[family] is not None:
                explicit_family[family] = tested_adjudication(
                    family, optional[family].get("decision"), optional_receipts[family],
                    tested_scope=optional[family].get("claim_boundary", "DECLARED_ARTIFACT_SCOPE"),
                    fallback_unresolved=f"{family} artifact has no terminal decision.",
                )
            elif isinstance(embedded, dict):
                explicit_family[family] = tested_adjudication(
                    family, embedded.get("decision"),
                    {"tournament_seal_sha256": tournament["seal_sha256"],
                     "family_record": embedded},
                    tested_scope="TOURNAMENT_DECLARED_F0_F1_SCOPE",
                    fallback_unresolved=f"Tournament {family} record has no terminal decision.",
                )
            if explicit_family.get(family, {}).get("state") == "TESTED_PROMOTED":
                survivors.append(family)
        for family in ("N3", "N5"):
            if family in explicit_family:
                nonlinear[family] = explicit_family[family]
            else:
                nonlinear[family] = optional_family(
                    family, optional[family], optional_receipts[family], survivors,
                    prerequisites[family],
                )
        oneoffs = {
            "M1": adjudicate_m1(
                m1, m1_receipt, optional["M1_DOWNSTREAM"], optional_receipts["M1_DOWNSTREAM"],
            ),
            "M2": adjudicate_m2(
                m2, m2_receipt, optional["M2_PHYSICAL"], optional_receipts["M2_PHYSICAL"],
                survivors,
            ),
            "M3": adjudicate_m3(
                optional["M3"], optional_receipts["M3"], tournament, rows,
            ),
            "M4": adjudicate_m4(optional["M4"], optional_receipts["M4"], survivors),
            "M5": adjudicate_m5(m5, m5_receipt, survivors, m5_rows),
            "M6": adjudicate_m6(
                optional["M6"], optional_receipts["M6"], tournament, rows,
            ),
            "M7": adjudicate_m7(m7, m7_receipt, survivors),
        }
        parent, parent_evidence = parent_receipt(runtime)
        auxiliary = validate_auxiliary_payloads(m1, m2)
        unresolved = [
            key for key, value in {**nonlinear, **oneoffs}.items()
            if value["state"] == "UNRESOLVED_JUSTIFIED_BRANCH"
        ]
        evidence = {
            "contextual_seam": seam_receipt, "tournament": tournament_receipt,
            "M1": m1_receipt, "M2": m2_receipt, "M5": m5_receipt,
            "M7": m7_receipt, "hooks": hooks_receipt,
            "optional": optional_receipts, "current_parent": parent_evidence,
            "embedded_file_receipts_verified": sorted(
                receipt_cache.values(), key=lambda value: value["path"],
            ),
        }
        fingerprint = hashlib.sha256(canonical({
            "artifact_seals": {
                key: value.get("seal_sha256") for key, value in evidence.items()
                if isinstance(value, dict)
            },
            "optional_seals": {
                key: value.get("seal_sha256") for key, value in optional_receipts.items()
            },
            "payload_hashes": [row["physical_payload"]["sha256"] for row in rows],
            "parent_payload_sha256": parent_evidence["payload"]["sha256"],
        })).hexdigest()
        outcome = (
            "OUTCOME_C_TESTED_NONLINEAR_REGION_CLOSED_ORACLE_INFORMATION_REMAINS_UNPHYSICALIZED"
            if not unresolved and not survivors else
            ("OUTCOME_A_OR_B_REQUIRES_F2_REPLICATION" if survivors else "OUTCOME_E_UNRESOLVED_BRANCH")
        )
        proven = ["N1/N2/N4/N6 were physically serialized and tested on contextual F1."]
        if not survivors:
            proven.append("No contextual nonlinear/native row survived its sealed F1 decision.")
        if oneoffs["M6"]["state"] == "TESTED_REJECTED":
            proven.append("The tested zero-Doctor native allocation failed contextual F1.")
        if oneoffs["M1"]["state"] == "TESTED_REJECTED":
            proven.append("M1 produced no row meeting its preregistered CI admission threshold.")
        suggested = []
        if m2 is not None and m2.get("skillful_trigger_rates"):
            suggested.append(
                "Compact conditional structure may exist because the M2 gate predicts high-error events."
            )
        if m1 is not None:
            suggested.append(
                "Useful oracle correction is concentrated but was not converted into a deployable survivor."
            )
        doctor_conclusion = (
            "Zero-Doctor native rows were tested and retired; Doctor remains justified for this "
            "local parent, not proven globally optimal."
            if oneoffs["M6"]["state"] == "TESTED_REJECTED" else
            "Doctor-versus-native remains governed by the M6 adjudication; no global optimum is claimed."
        )
        byte_auction = seal({
            "schema": "hawking.kimi_k26.final_byte_auction.v1",
            "status": "PASS" if not unresolved else "INCOMPLETE",
            "sealed_at": now(), "closure_fingerprint_sha256": fingerprint,
            "physical_rate_law": {
                "logical_weights": LOGICAL_WEIGHTS,
                "complete_ceiling_bytes": COMPLETE_CEILING_BYTES,
                "complete_ceiling_bpw": COMPLETE_CEILING_BYTES * 8 / LOGICAL_WEIGHTS,
                "all_payloads_hash_and_size_verified": True,
            },
            "current_best": {
                "candidate": "P1_DUAL_PATH_RECOVERY_R16X2",
                "complete_physical_bytes": parent_evidence["payload"]["bytes"],
                "complete_bpw": CURRENT_PARENT_BPW,
                "allocation": {
                    "base_component_bytes": parent["payload"]["base_component_bytes"],
                    "doctor_component_bytes": parent["payload"]["doctor_component_bytes"],
                    "header_overhead_bytes": parent["payload"]["header_overhead_bytes"],
                    "unused_ceiling_bytes": COMPLETE_CEILING_BYTES - CURRENT_PARENT_BYTES,
                },
                "payload": parent_evidence["payload"], "f2_promotable": False,
            },
            "tested_physical_candidates": rows,
            "rate_stress_candidates": m5_rows,
            "nonlinear_family_adjudication": nonlinear,
            "nondeployable_oracle_and_gate_accounting": auxiliary,
            "doctor_versus_native": {
                "doctor_parent_bytes": parent["payload"]["doctor_component_bytes"],
                "zero_doctor_native_rows": [
                    row for row in rows
                    if row["family"] == "N6" and row["physical_payload"].get("doctor_component_bytes") == 0
                ],
                "conclusion": doctor_conclusion,
            },
            "decision": (
                "RETAIN_P1_DUAL_PATH_AS_CURRENT_BEST" if not survivors else
                "F1_SURVIVOR_REQUIRES_F2_BEFORE_BYTE_AUCTION_WIN"
            ),
            "unresolved_branches": unresolved, "evidence": evidence,
        })
        closure = seal({
            "schema": "hawking.kimi_k26.gravity_oneoff_closure.v1",
            "status": "PASS" if not unresolved else "INCOMPLETE",
            "experiment_id": "GRAVITY_N3_N5_M1_M7_TERMINAL_ADJUDICATION",
            "sealed_at": now(), "closure_fingerprint_sha256": fingerprint,
            "claim_boundary": (
                "KIMI LOCAL/CONTEXTUAL EXPERT0 NONLINEAR F0/F1 PLUS CACHED ONE-OFFS; "
                "not a universal compression impossibility proof"
            ),
            "contextual_f1_survivors": survivors,
            "nonlinear_families": nonlinear, "oneoffs": oneoffs,
            "classification_law": {
                "TESTED_REJECTED": "a sealed physical experiment ran and failed its terminal gate",
                "PREREQUISITE_ABSENT": "the experiment did not run because its named scientific parent is absent",
                "UNRESOLVED_JUSTIFIED_BRANCH": "required evidence remains and closure is forbidden",
            },
            "unresolved_branches": unresolved,
            "decision": (
                "ALL_JUSTIFIED_BRANCHES_TERMINAL_READY_FOR_FINAL_REPORT" if not unresolved else
                "DO_NOT_CLOSE_UNRESOLVED_JUSTIFIED_BRANCHES"
            ),
            "terminal_outcome_recommendation": outcome,
            "proven": proven,
            "suggested": suggested,
            "not_proven": [
                "No nonlinear representation can ever work for Kimi.",
                "Rates below 0.75 BPW are intrinsically impossible when M5 is prerequisite-absent.",
                "Doctor is globally optimal.",
                "The nonlinear oracle gap is zero; it is undefined when no physical nonlinear survivor exists.",
            ],
            "next_action": (
                "WRITE_KIMI_GRAVITY_FINAL_AND_NEXT_PARENT_TRANSFER" if not unresolved else unresolved[0]
            ),
            "byte_auction_seal_sha256": byte_auction["seal_sha256"],
            "evidence": evidence,
        })
        mirror_artifact(repo, runtime, BYTE_AUCTION, byte_auction)
        mirror_artifact(repo, runtime, CLOSURE, closure)
        ledger_record = append_mirrored_ledger(repo, runtime, {
            "event": "GRAVITY_ONEOFF_CLOSURE", "at": now(),
            "experiment_id": closure["experiment_id"],
            "hypothesis": "Every justified nonlinear family and bounded one-off has a terminal evidence class.",
            "decision": closure["decision"],
            "metrics": {
                "contextual_f1_survivors": survivors,
                "nonlinear_states": {key: value["state"] for key, value in nonlinear.items()},
                "oneoff_states": {key: value["state"] for key, value in oneoffs.items()},
                "unresolved_branches": unresolved,
            },
            "byte_auction_seal_sha256": byte_auction["seal_sha256"],
            "evidence_seal_sha256": closure["seal_sha256"],
            "next_run_rationale": closure["next_action"],
            "faults": [],
        }, fingerprint)
        status = write_mirrored_status(repo, runtime, closure, ledger_record)
        print(json.dumps({
            "status": closure["status"], "decision": closure["decision"],
            "terminal_outcome_recommendation": outcome,
            "unresolved_branches": unresolved,
            "byte_auction_seal_sha256": byte_auction["seal_sha256"],
            "closure_seal_sha256": closure["seal_sha256"],
            "status_seal_sha256": status["seal_sha256"],
        }, sort_keys=True))
        return 0 if not unresolved else 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
