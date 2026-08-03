#!/usr/bin/env python3.12
"""Known-failure registry: sealed negatives with cheap runnable reproductions.

Each entry records a campaign negative result and a check that would catch a
silent regression of the *claim* (not necessarily the full original compute).
A later stage that "fixes" a sealed negative by forgetting it fails the check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lab.layout import find_evidence
from tools.odyssey._paths import ROOT, T0_DIR

SCHEMA = "hawking.odyssey.t0.known_failures.v1"
REGISTRY_PATH = T0_DIR / "KNOWN_FAILURES_REGISTRY.json"


def _load_json(rel: str) -> dict[str, Any] | None:
    # Callers name receipts by basename and that same string is the "source"
    # field of every registry entry, so it stays a basename; only the lookup
    # follows the artifact into workspace/campaign/evidence/<area>/<campaign>/.
    path = ROOT / rel
    if not path.is_file():
        path = find_evidence(rel) or path
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def check_expert_wave_negative() -> dict[str, Any]:
    """Sealed: expert-wave is correct but slower; not promoted; flag default off."""
    doc = _load_json("HAWKING_EXPERT_WAVE_NEGATIVE.json")
    if doc is None:
        return {"id": "expert_wave_negative", "status": "FAIL", "reason": "receipt missing"}
    verdict = str(doc.get("verdict", ""))
    ratio = (doc.get("measured") or {}).get("ratio")
    # Claim: NEGATIVE, ratio < 1.0 (slower), not promoted.
    ok = (
        "NEGATIVE" in verdict.upper()
        and ratio is not None
        and float(ratio) < 1.0
        and "NOT promoted" in verdict
    )
    return {
        "id": "expert_wave_negative",
        "status": "PASS" if ok else "FAIL",
        "source": "HAWKING_EXPERT_WAVE_NEGATIVE.json",
        "claim": "Expert-wave is correct/gated but slower (ratio<1); not promoted; flag stays off.",
        "observed": {"verdict_prefix": verdict[:80], "ratio": ratio},
        "expected_failing_outcome": "ratio < 1.0 and NEGATIVE / not promoted",
        "reproduction": "parse sealed receipt; assert measured.ratio < 1 and not-promoted language",
    }


def check_expert_wave_rejected() -> dict[str, Any]:
    doc = _load_json("HAWKING_EXPERT_WAVE_REJECTED.json")
    if doc is None:
        return {"id": "expert_wave_rejected", "status": "FAIL", "reason": "receipt missing"}
    verdict = str(doc.get("verdict", ""))
    why = doc.get("why_it_was_rejected") or []
    ok = "REJECTED" in verdict.upper() and len(why) >= 1
    return {
        "id": "expert_wave_rejected",
        "status": "PASS" if ok else "FAIL",
        "source": "HAWKING_EXPERT_WAVE_REJECTED.json",
        "claim": "Expert-wave merge was REJECTED AND REVERTED after parity-gate failures.",
        "observed": {"verdict": verdict[:100], "n_rejection_reasons": len(why)},
        "expected_failing_outcome": "verdict contains REJECTED; at least one rejection reason",
        "reproduction": "parse sealed rejection receipt",
    }


def check_resident_state_negative() -> dict[str, Any]:
    doc = _load_json("HAWKING_RESIDENT_STATE_NEGATIVE.json")
    if doc is None:
        return {"id": "resident_state_negative", "status": "FAIL", "reason": "receipt missing"}
    result = str(doc.get("result", ""))
    ok = "NEGATIVE" in result.upper() and "no measurable speedup" in result.lower()
    return {
        "id": "resident_state_negative",
        "status": "PASS" if ok else "FAIL",
        "source": "HAWKING_RESIDENT_STATE_NEGATIVE.json",
        "claim": "GPU-resident state: no measurable speedup (parity holds, performance does not).",
        "observed": {"result_prefix": result[:120]},
        "expected_failing_outcome": "result is NEGATIVE / no measurable speedup",
        "reproduction": "parse sealed resident-state receipt",
    }


def check_r0_geometry_ceiling() -> dict[str, Any]:
    doc = _load_json("GLM52_R1_GEOMETRY_INVALID_FINDING.json")
    if doc is None:
        return {"id": "r0_geometry_ceiling", "status": "FAIL", "reason": "receipt missing"}
    summary = str(doc.get("summary", ""))
    evidence = doc.get("evidence") or {}
    candidates = evidence.get("candidates_tested") or []
    # Claim: no admissible candidate strictly between R0 0.875 and 1.0.
    r0_ok = any(
        c.get("dim") == 8 and c.get("k") == 128 and c.get("admissible_routed_expert") is True
        for c in candidates
    )
    # Any candidate with 0.875 < bpw < 1.0 that claims admissible on routed_expert would regress.
    illegal_admitted = [
        c
        for c in candidates
        if 0.875 < float(c.get("nominal_bpw", 0)) < 1.0
        and c.get("admissible_routed_expert") is True
    ]
    ok = r0_ok and not illegal_admitted and "ceiling" in summary.lower()
    return {
        "id": "r0_geometry_ceiling",
        "status": "PASS" if ok else "FAIL",
        "source": "GLM52_R1_GEOMETRY_INVALID_FINDING.json",
        "claim": "R0 is the practical PQ ceiling; no admissible rate strictly between 0.875 and 1.0.",
        "observed": {
            "r0_present_admissible": r0_ok,
            "illegal_mid_rate_admitted": illegal_admitted,
            "summary_prefix": summary[:120],
        },
        "expected_failing_outcome": "R0 admissible; no mid-rate candidate admissible",
        "reproduction": "re-evaluate candidates_tested table for mid-rate false admissions",
    }


def check_functional_student_cascade() -> dict[str, Any]:
    doc = _load_json("GLM52_FUNCTIONAL_DECISION.json")
    if doc is None:
        return {"id": "functional_student_cascade", "status": "FAIL", "reason": "receipt missing"}
    closure = doc.get("closure") or {}
    # Cascade is expansive (not contractive) — the sealed negative on composition.
    all_expansive = closure.get("all_expansive_at_every_magnitude")
    cascade = closure.get("cascade_final_skill") or {}
    ok = all_expansive is True and isinstance(cascade, dict) and len(cascade) >= 1
    # Also require the student contract still marks FUNCTIONAL_PARTIAL_ONLY / not full replace.
    contract = _load_json("GLM52_FUNCTIONAL_STUDENT_CONTRACT.json")
    contract_ok = contract is not None and "FUNCTIONAL_PARTIAL" in str(contract.get("status", ""))
    return {
        "id": "functional_student_cascade",
        "status": "PASS" if ok and contract_ok else "FAIL",
        "source": "GLM52_FUNCTIONAL_DECISION.json + GLM52_FUNCTIONAL_STUDENT_CONTRACT.json",
        "claim": (
            "Functional student cascade is expansive at every magnitude; held under "
            "FUNCTIONAL_PARTIAL_ONLY — not a full MoE replacement success."
        ),
        "observed": {
            "all_expansive_at_every_magnitude": all_expansive,
            "cascade_final_skill": cascade,
            "student_contract_status": (contract or {}).get("status"),
        },
        "expected_failing_outcome": "all_expansive true; student contract FUNCTIONAL_PARTIAL_ONLY",
        "reproduction": "parse sealed functional decision + student contract",
    }


CHECKS: list[Callable[[], dict[str, Any]]] = [
    check_expert_wave_negative,
    check_expert_wave_rejected,
    check_resident_state_negative,
    check_r0_geometry_ceiling,
    check_functional_student_cascade,
]


def build_registry() -> dict[str, Any]:
    entries = [fn() for fn in CHECKS]
    return {
        "schema": SCHEMA,
        "status": "PASS" if all(e["status"] == "PASS" for e in entries) else "FAIL",
        "n_entries": len(entries),
        "entries": entries,
        "what_was_checked": [e["id"] for e in entries],
        "what_was_skipped": [
            "full re-benchmark of expert-wave and resident-state wall-clock (GPU lane; heavy)",
            "full functional cascade recompute (sealed measured results stand)",
        ],
        "note": (
            "Reproductions assert the sealed claim still reads as a negative. They do not "
            "re-run multi-hour GPU experiments. A silent rewrite of a sealed negative fails."
        ),
    }


def write_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    reg = build_registry()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    return reg


def main(argv: list[str] | None = None) -> int:
    reg = write_registry()
    print(json.dumps(reg, indent=2, sort_keys=True))
    return 0 if reg["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    raise SystemExit(main())
