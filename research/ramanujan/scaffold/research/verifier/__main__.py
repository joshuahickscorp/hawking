"""CLI: probe backends and run the expert-iteration fixture end-to-end.

  python3 -m ramanujan.verifier probe
  python3 -m ramanujan.verifier fixture
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ramanujan.layout import FIXTURES_ROOT
from ramanujan.verifier.base import VerificationRequest, Verdict
from ramanujan.verifier.expert_iteration import (
    ExpertIterationHarness,
    fixture_wrong_then_right_student,
)
from ramanujan.verifier.registry import default_registry, probe_backends


def cmd_probe() -> int:
    report = probe_backends()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("has_real_backend") else 2


def cmd_fixture(path: Path | None = None) -> int:
    fixture_path = path or (FIXTURES_ROOT / "expert_iteration_math.json")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    registry = default_registry()
    results = []
    failures = 0

    for problem in data.get("problems") or []:
        wrong = str(problem["wrong_attempt"])
        right = str(problem["correct_repair"])
        student, repair = fixture_wrong_then_right_student(wrong, right)
        harness = ExpertIterationHarness(
            student=student,
            repair=repair,
            registry=registry,
            max_repairs=1,
        )
        out = harness.run(problem)
        # Prove wrong attempt was caught.
        first = out.rounds[0]["verification"]
        if first["verdict"] != Verdict.REJECTED.value:
            failures += 1
            results.append(
                {
                    "id": problem["id"],
                    "ok": False,
                    "reason": f"expected first attempt REJECTED, got {first['verdict']}",
                }
            )
            continue
        if not out.accepted or not out.trajectory.get("admitted"):
            failures += 1
            results.append(
                {
                    "id": problem["id"],
                    "ok": False,
                    "reason": f"repair not accepted: {out.stop_reason}",
                }
            )
            continue
        if out.paired_trace.get("admitted") is not True:
            failures += 1
            results.append(
                {
                    "id": problem["id"],
                    "ok": False,
                    "reason": "paired_trace not admitted",
                }
            )
            continue
        # Prove a still-wrong repair would not be accepted.
        bad_student, bad_repair = fixture_wrong_then_right_student(wrong, wrong)
        bad = ExpertIterationHarness(
            student=bad_student,
            repair=bad_repair,
            registry=registry,
            max_repairs=1,
        ).run(problem)
        if bad.accepted or bad.trajectory.get("admitted"):
            failures += 1
            results.append(
                {
                    "id": problem["id"],
                    "ok": False,
                    "reason": "unverified/wrong repair was incorrectly admitted",
                }
            )
            continue
        results.append(
            {
                "id": problem["id"],
                "ok": True,
                "wrong_caught": True,
                "repair_accepted": True,
                "bad_repair_rejected": True,
                "trajectory_id": out.trajectory.get("id"),
                "paired_trace_id": out.paired_trace.get("id"),
                "content_hash": out.trajectory.get("content_hash"),
                "teacher_critique_gate": (out.teacher_critique or {}).get("status"),
            }
        )

    # Lean fail-closed smoke.
    lean = data.get("lean_smoke") or {}
    if lean:
        req = VerificationRequest(
            problem_id=str(lean.get("id")),
            statement=str(lean.get("statement")),
            kind=str(lean.get("kind", "lean_capsule")),
            claimed_answer=str(lean.get("proof_lean", "")),
            payload={"proof_lean": lean.get("proof_lean", "")},
        )
        lean_result = registry.verify(req)
        lean_ok = lean_result.verdict.value == lean.get("expect_verdict", "REJECTED")
        if not lean_ok:
            failures += 1
        results.append(
            {
                "id": lean.get("id"),
                "ok": lean_ok,
                "verdict": lean_result.verdict.value,
                "detail": lean_result.detail,
            }
        )

    report = {
        "schema": "hawking.ramanujan.expert_iteration_fixture_run.v1",
        "fixture": str(fixture_path),
        "backend_probe": registry.probe_report(),
        "results": results,
        "failures": failures,
        "status": "PASS" if failures == 0 else "FAIL",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ramanujan verifier / expert-iteration CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="report REAL vs GATED vs ABSENT backends")
    p_fix = sub.add_parser("fixture", help="run small exact-numeric expert-iteration fixture")
    p_fix.add_argument("--fixture", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "probe":
        return cmd_probe()
    if args.cmd == "fixture":
        return cmd_fixture(args.fixture)
    return 2


if __name__ == "__main__":
    sys.exit(main())
