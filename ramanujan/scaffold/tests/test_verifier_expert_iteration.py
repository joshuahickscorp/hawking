"""Tests for verifier backends + expert-iteration loop.

Hard properties:
* wrong attempt is REJECTED by the real exact_numeric backend
* repair is admitted only when verification ACCEPTs
* wrong repair is never admitted
* Lean with sorry is REJECTED (static fail-closed)
* teacher critique without GLM is REQUIRES_GLM_ACCESS (not faked)
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from ramanujan.layout import FIXTURES_ROOT
from ramanujan.verifier.base import (
    BackendStatus,
    VerificationRequest,
    Verdict,
)
from ramanujan.verifier.exact_numeric import ExactNumericBackend, evaluate_exact
from ramanujan.verifier.expert_iteration import (
    REQUIRES_GLM_ACCESS,
    ExpertIterationHarness,
    StudentAttempt,
    fixture_wrong_then_right_student,
    request_teacher_critique,
)
from ramanujan.verifier.lean_backend import LeanBackend
from ramanujan.verifier.registry import VerifierRegistry, probe_backends
from ramanujan.verifier.trajectory import (
    PAIRED_TRACE_SCHEMA,
    VERIFIED_TRAJECTORY_SCHEMA,
    emit_paired_trace_record,
)


FIXTURE = FIXTURES_ROOT / "expert_iteration_math.json"


class ExactNumericTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = ExactNumericBackend()

    def test_availability_is_real(self) -> None:
        avail = self.backend.availability()
        self.assertEqual(avail.status, BackendStatus.REAL)

    def test_two_plus_two_accepts(self) -> None:
        req = VerificationRequest(
            problem_id="t",
            statement="2+2",
            kind="exact_numeric",
            claimed_answer="4",
            payload={"expression": "2 + 2"},
        )
        result = self.backend.verify(req)
        self.assertEqual(result.verdict, Verdict.ACCEPTED)

    def test_two_plus_two_rejects_five(self) -> None:
        req = VerificationRequest(
            problem_id="t",
            statement="2+2",
            kind="exact_numeric",
            claimed_answer="5",
            payload={"expression": "2 + 2"},
        )
        result = self.backend.verify(req)
        self.assertEqual(result.verdict, Verdict.REJECTED)

    def test_fraction_exact(self) -> None:
        self.assertEqual(evaluate_exact("1/2 + 1/3"), evaluate_exact("5/6"))
        req = VerificationRequest(
            problem_id="t",
            statement="1/2+1/3",
            kind="exact_numeric",
            claimed_answer="5/6",
            payload={"expression": "1/2 + 1/3"},
        )
        self.assertTrue(self.backend.verify(req).accepted)

    def test_rejects_code_injection(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_exact("__import__('os').system('echo pwned')")


class LeanFailClosedTests(unittest.TestCase):
    def test_sorry_is_rejected(self) -> None:
        backend = LeanBackend()
        req = VerificationRequest(
            problem_id="lean:sorry",
            statement="2+2=4",
            kind="lean_capsule",
            claimed_answer="by sorry",
            payload={"proof_lean": "theorem t : (2:Nat)+2 = 4 := by sorry"},
        )
        result = backend.verify(req)
        self.assertEqual(result.verdict, Verdict.REJECTED)
        self.assertIn("sorry", result.detail)

    def test_clean_looking_proof_without_container_is_unavailable(self) -> None:
        backend = LeanBackend()
        req = VerificationRequest(
            problem_id="lean:rfl",
            statement="True",
            kind="lean_capsule",
            claimed_answer="rfl",
            payload={"proof_lean": "theorem t : True := trivial"},
        )
        result = backend.verify(req)
        # Must not fake ACCEPTED without machine-check.
        self.assertNotEqual(result.verdict, Verdict.ACCEPTED)
        self.assertIn(result.verdict, {Verdict.UNAVAILABLE, Verdict.REJECTED})


class TeacherCritiqueGateTests(unittest.TestCase):
    def test_critique_requires_glm_access(self) -> None:
        os.environ.pop("HAWKING_GLM_TEACHER_ACCESS", None)
        from ramanujan.verifier.base import VerificationResult

        gate = request_teacher_critique(
            statement="2+2",
            failed_answer="5",
            verification=VerificationResult(
                backend_id="exact_numeric",
                verdict=Verdict.REJECTED,
                detail="wrong",
            ),
        )
        self.assertFalse(gate.available)
        self.assertEqual(gate.status, REQUIRES_GLM_ACCESS)
        self.assertIsNone(gate.critique)


class ExpertIterationTests(unittest.TestCase):
    def test_wrong_caught_and_repair_accepted(self) -> None:
        student, repair = fixture_wrong_then_right_student("5", "4")
        harness = ExpertIterationHarness(
            student=student,
            repair=repair,
            registry=VerifierRegistry([ExactNumericBackend()]),
            max_repairs=1,
        )
        problem = {
            "id": "fix:two_plus_two",
            "kind": "exact_numeric",
            "statement": "What is 2 + 2?",
            "expression": "2 + 2",
        }
        out = harness.run(problem)
        self.assertEqual(out.rounds[0]["verification"]["verdict"], Verdict.REJECTED.value)
        self.assertTrue(out.accepted)
        self.assertTrue(out.trajectory["admitted"])
        self.assertEqual(out.trajectory["schema"], VERIFIED_TRAJECTORY_SCHEMA)
        self.assertEqual(out.trajectory["final_answer"], "4")
        self.assertEqual(out.trajectory["disposition"], "EXACTLY_REPRODUCED")
        self.assertTrue(out.paired_trace["admitted"])
        self.assertEqual(out.paired_trace["schema"], PAIRED_TRACE_SCHEMA)
        self.assertEqual(out.paired_trace["input"]["answer"], "5")
        self.assertEqual(out.paired_trace["target"]["answer"], "4")
        # Teacher critique gated, not faked.
        self.assertEqual(out.teacher_critique["status"], REQUIRES_GLM_ACCESS)

    def test_wrong_repair_never_admitted(self) -> None:
        student, repair = fixture_wrong_then_right_student("5", "5")
        harness = ExpertIterationHarness(
            student=student,
            repair=repair,
            registry=VerifierRegistry([ExactNumericBackend()]),
            max_repairs=1,
        )
        out = harness.run(
            {
                "id": "fix:two_plus_two",
                "kind": "exact_numeric",
                "statement": "What is 2 + 2?",
                "expression": "2 + 2",
            }
        )
        self.assertFalse(out.accepted)
        self.assertFalse(out.trajectory["admitted"])
        self.assertIsNone(out.trajectory["final_answer"])
        self.assertFalse(out.paired_trace["admitted"])

    def test_unverified_never_admitted_even_if_student_confident(self) -> None:
        def student(_p):
            return StudentAttempt(answer="maybe 4?", plan=("guess",))

        harness = ExpertIterationHarness(
            student=student,
            repair=None,
            registry=VerifierRegistry([ExactNumericBackend()]),
            max_repairs=0,
        )
        out = harness.run(
            {
                "id": "fix:x",
                "kind": "exact_numeric",
                "statement": "2+2",
                "expression": "2 + 2",
            }
        )
        self.assertFalse(out.accepted)
        self.assertFalse(out.trajectory["admitted"])

    def test_full_fixture_file(self) -> None:
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        registry = VerifierRegistry([ExactNumericBackend(), LeanBackend()])
        for problem in data["problems"]:
            student, repair = fixture_wrong_then_right_student(
                problem["wrong_attempt"], problem["correct_repair"]
            )
            out = ExpertIterationHarness(
                student=student,
                repair=repair,
                registry=registry,
                max_repairs=1,
            ).run(problem)
            self.assertEqual(
                out.rounds[0]["verification"]["verdict"],
                Verdict.REJECTED.value,
                msg=problem["id"],
            )
            self.assertTrue(out.accepted, msg=problem["id"])
            self.assertTrue(out.trajectory["admitted"], msg=problem["id"])
            # still-wrong path
            bad_s, bad_r = fixture_wrong_then_right_student(
                problem["wrong_attempt"], problem["wrong_attempt"]
            )
            bad = ExpertIterationHarness(
                student=bad_s, repair=bad_r, registry=registry, max_repairs=1
            ).run(problem)
            self.assertFalse(bad.accepted, msg=problem["id"])
            self.assertFalse(bad.trajectory["admitted"], msg=problem["id"])

    def test_paired_trace_not_admitted_without_verify(self) -> None:
        traj = {
            "id": "vt:x",
            "problem_id": "x",
            "statement": "2+2",
            "statement_hash": "a" * 64,
            "admitted": False,
            "disposition": "REJECTED",
            "attempts": [{"round": 0, "answer": "5", "verdict": "REJECTED"}],
            "final_answer": None,
            "verifier_outcomes": [],
            "plan": [],
            "subgoals": [],
            "formal_states": [],
            "actions": [],
            "tool_calls": [],
            "provenance": {},
        }
        paired = emit_paired_trace_record(traj, wrong_attempt="5", verified_answer="4")
        self.assertFalse(paired["admitted"])
        self.assertIsNone(paired["target"]["answer"])


class ProbeTests(unittest.TestCase):
    def test_probe_reports_exact_numeric_real(self) -> None:
        report = probe_backends()
        self.assertIn("exact_numeric", report["real"])
        self.assertTrue(report["has_real_backend"])
        # Lean must not claim REAL machine-check without full container proof path;
        # it may be GATED or ABSENT depending on host.
        for row in report["backends"]:
            if row["backend_id"] == "lean":
                self.assertIn(row["status"], {"GATED", "ABSENT", "REAL"})
                # Even if somehow REAL, verify() still fails closed without capsule —
                # the important invariant is we never invent ACCEPTED for sorry.
                break


if __name__ == "__main__":
    unittest.main()
