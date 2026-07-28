#!/usr/bin/env python3.12
"""Deterministic tests for the final-ascent control plane publisher.

    python3.12 -m tools.campaign.test_final_ascent_status
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.campaign import final_ascent_status as fas


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write(path, json.dumps(value, indent=2) + "\n")


def _minimal_repo(tmp: Path, *, approved: bool = False, gen_b_refused: bool = True) -> Path:
    """Synthetic evidence tree — never invents live process state."""
    (tmp / "odyssey" / "launch").mkdir(parents=True)
    _write(tmp / "odyssey" / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED", "false\n")
    substrates = []
    if approved:
        substrates.append({
            "name": "Math-Preserve-v2-test",
            "artifact_index_sha256": "a" * 64,
            "capability_verdict": "APPROVED",
            "capability_reason": "test fixture",
        })
    else:
        substrates.append({
            "name": "GLM-5.2-H0.98-Math-Preserve",
            "artifact_index_sha256": "b" * 64,
            "capability_verdict": "REFUSED",
            "capability_reason": "test fixture collapse",
        })
    _write_json(
        tmp / "odyssey" / "launch" / "SUBSTRATE_CAPABILITY.json",
        {
            "schema": "hawking.odyssey.substrate_capability.v1",
            "substrates": substrates,
            "default_for_unlisted": "UNVERIFIED",
        },
    )
    if gen_b_refused:
        _write_json(
            tmp / "GLM52_GENERATION_B_CAPABILITY_VERDICT.json",
            {
                "schema": "hawking.substrate.capability_gate_run.v1",
                "capability_verdict": "REFUSED",
                "artifact_index_sha256": "c" * 64,
                "gates": [{"gate": "G_math", "status": "FAIL"}],
                "diagnosis": {"verdict": "REFUSED for tests"},
            },
        )
    _write(
        tmp / "HAWKING_RESUME_CHECKPOINT.md",
        "# resume\n\nQ0 ACHIEVED (capsule re-proves).\nRAMANUJAN_SANDBOX_READY is **not** reached.\n",
    )
    _write_json(
        tmp / "HAWKING_HEAVY_CONTINUATION_STATUS.json",
        {
            "endpoint": "RAMANUJAN_SANDBOX_READY",
            "endpoint_reached": False,
            "endpoint_blockers": ["Q0 ACHIEVED and verified by running the container"],
            "formal_environment": {"q0": "UNBLOCKED_FOR_LEAN", "status": "Lean REAL"},
        },
    )
    _write_json(tmp / "GLM52_BYTE_ATTRIBUTION.json", {"schema": "test"})
    _write_json(tmp / "HAWKING_BASE_TRUE_TPS.json", {"tps": 0.39})
    _write_json(tmp / "FABRIC_SOFTWARE_STATUS.json", {"status": "prep"})
    _write_json(tmp / "HAWKING_CONSOLIDATION_INVENTORY.json", {"status": "inventory"})
    # Directive continuation files intentionally absent.
    return tmp


class TestDagAcyclicity(unittest.TestCase):
    def test_static_edges_are_acyclic(self) -> None:
        nodes = [spec["id"] for spec in fas.LANE_SPECS]
        self.assertTrue(fas.dag_is_acyclic(nodes, fas.DAG_EDGES))

    def test_cycle_detected(self) -> None:
        self.assertFalse(
            fas.dag_is_acyclic(
                ["A", "B"],
                [{"from": "A", "to": "B"}, {"from": "B", "to": "A"}],
            )
        )

    def test_build_dag_marks_acyclic(self) -> None:
        dag = fas.build_dag("2026-01-01T00:00:00Z")
        self.assertTrue(dag["acyclic"])
        self.assertEqual(dag["schema"], "hawking.final_ascent.dependency_dag.v1")
        self.assertTrue(dag["generated"])
        self.assertTrue(dag["do_not_hand_edit"])


class TestSchemaCompleteness(unittest.TestCase):
    def test_built_status_has_required_lane_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td))
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        errors = fas.validate_status_schema(
            {k: v for k, v in status.items() if not k.startswith("_")}
        )
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(status["lanes"]), 12)
        fields = fas.required_lane_fields()
        for lane in status["lanes"]:
            for field in fields:
                self.assertIn(field, lane, msg=f"{lane['id']} missing {field}")


class TestFencePreservation(unittest.TestCase):
    def test_fences_stay_false_when_launch_file_false(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td))
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        for name in fas.FENCE_NAMES:
            self.assertIs(status["fences"][name], False, msg=name)

    def test_research_and_kernel_never_true_from_builder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td), approved=True)
            # Even if someone leaves a true launch file, research/kernel stay closed.
            _write(root / "odyssey" / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED", "true\n")
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        self.assertIs(status["fences"]["RAMANUJAN_RESEARCH_AUTHORIZED"], False)
        self.assertIs(status["fences"]["HIDE_KERNEL_TURN"], False)
        # Launch file may report true; builder does not rewrite the file (safety).
        self.assertIs(status["fences"]["ODYSSEY_LAUNCH_AUTHORIZED"], True)
        self.assertFalse(status["endpoint_reached"])


class TestCapabilityRefusal(unittest.TestCase):
    def test_absent_substrate_register_refuses_capability(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "odyssey" / "launch").mkdir(parents=True)
            _write(root / "odyssey" / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED", "false\n")
            # No SUBSTRATE_CAPABILITY.json
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        self.assertFalse(status["capability_gate"]["any_approved"])
        self.assertIn(status["capability_gate"]["summary"], {"ABSENT", "UNPARSEABLE", "NONE_LISTED"})
        odyssey = next(l for l in status["lanes"] if l["id"] == "FA05")
        math_frozen = next(l for l in status["lanes"] if l["id"] == "FA06")
        self.assertIn("BLOCKED", odyssey["status"])
        self.assertIn("BLOCKED", math_frozen["status"])
        self.assertFalse(status["endpoint_reached"])

    def test_refused_generation_b_blocks_odyssey_and_math_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td), approved=False, gen_b_refused=True)
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        self.assertFalse(status["capability_gate"]["any_approved"])
        fa02 = next(l for l in status["lanes"] if l["id"] == "FA02")
        fa05 = next(l for l in status["lanes"] if l["id"] == "FA05")
        fa06 = next(l for l in status["lanes"] if l["id"] == "FA06")
        fa04 = next(l for l in status["lanes"] if l["id"] == "FA04")
        self.assertEqual(fa02["status"], "CAPABILITY_REFUSED")
        self.assertEqual(fa05["status"], "BLOCKED_CAPABILITY_REFUSED")
        self.assertEqual(fa06["status"], "BLOCKED_CAPABILITY_REFUSED")
        self.assertIn("KERNEL_TURN", fa04["status"])
        self.assertIn("REFUSED", status["why"])

    def test_approved_substrate_unblocks_capability_wall_but_not_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td), approved=True)
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        self.assertTrue(status["capability_gate"]["any_approved"])
        fa05 = next(l for l in status["lanes"] if l["id"] == "FA05")
        self.assertNotEqual(fa05["status"], "BLOCKED_CAPABILITY_REFUSED")
        self.assertFalse(status["endpoint_reached"])


class TestIdempotentLedger(unittest.TestCase):
    def test_second_publish_does_not_duplicate_transition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td))
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        # Freeze timestamps so shape comparison is the only variable.
                        with mock.patch.object(fas, "now", return_value="2026-07-28T00:00:00Z"):
                            s1 = fas.build(root)
                            r1 = fas.publish(s1, root)
                            s2 = fas.build(root)
                            r2 = fas.publish(s2, root)
            ledger = (root / fas.LEDGER).read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(r1["ledger_appended"])
            self.assertFalse(r2["ledger_appended"])
            self.assertEqual(len(ledger), 1)

    def test_status_change_appends_new_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td), approved=False)
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        with mock.patch.object(fas, "now", return_value="2026-07-28T00:00:00Z"):
                            s1 = fas.build(root)
                            fas.publish(s1, root)
                        # Flip capability evidence and republish.
                        _write_json(
                            root / "odyssey" / "launch" / "SUBSTRATE_CAPABILITY.json",
                            {
                                "substrates": [{
                                    "name": "v2",
                                    "artifact_index_sha256": "d" * 64,
                                    "capability_verdict": "APPROVED",
                                }],
                            },
                        )
                        with mock.patch.object(fas, "now", return_value="2026-07-28T01:00:00Z"):
                            s2 = fas.build(root)
                            r2 = fas.publish(s2, root)
            ledger = (root / fas.LEDGER).read_text(encoding="utf-8").strip().splitlines()
            self.assertTrue(r2["ledger_appended"])
            self.assertEqual(len(ledger), 2)


class TestAtomicPublication(unittest.TestCase):
    def test_atomic_write_replaces_and_leaves_no_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "out.json"
            fas.atomic_write_json(path, {"a": 1})
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text())["a"], 1)
            fas.atomic_write_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text())["a"], 2)
            leftovers = list(Path(td).glob(".*.tmp"))
            self.assertEqual(leftovers, [])

    def test_publish_writes_all_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td))
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
                        result = fas.publish(status, root)
            required = [
                fas.STATUS_MD,
                fas.STATUS_JSON,
                fas.LEDGER,
                fas.DAG,
                fas.OWNERSHIP,
                fas.GOAL,
                fas.NEXT_COMMAND,
            ]
            for name in required:
                self.assertTrue((root / name).is_file(), msg=name)
            self.assertEqual(set(result["written"]), set(required))
            md = (root / fas.STATUS_MD).read_text(encoding="utf-8")
            self.assertIn("Do not hand-edit", md)
            self.assertIn("Generated from live evidence", md)
            goal = (root / fas.GOAL).read_text(encoding="utf-8")
            self.assertIn("Do not hand-edit", goal)
            for absent in fas.ABSENT_DIRECTIVE_FILES:
                self.assertIn(absent, goal)
            ownership = json.loads((root / fas.OWNERSHIP).read_text(encoding="utf-8"))
            self.assertTrue(ownership["generated"])
            self.assertTrue(ownership["do_not_hand_edit"])
            dag = json.loads((root / fas.DAG).read_text(encoding="utf-8"))
            self.assertTrue(fas.dag_is_acyclic(list(dag["nodes"]), dag["edges"]))
            # next command is executable diagnose script
            next_sh = root / fas.NEXT_COMMAND
            self.assertTrue(os.access(next_sh, os.X_OK))
            body = next_sh.read_text(encoding="utf-8")
            self.assertIn("diagnose", body)
            self.assertIn("refuse_stale_lease", body)


class TestQ0AndAbsenceHonesty(unittest.TestCase):
    def test_q0_achieved_from_resume_but_q1q6_false(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td))
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        self.assertTrue(status["q0"]["q0_achieved"])
        self.assertFalse(status["q0"]["q1_q6_achieved"])
        fa12 = next(l for l in status["lanes"] if l["id"] == "FA12")
        self.assertIn("Q0_ACHIEVED", fa12["status"])

    def test_absent_directive_files_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _minimal_repo(Path(td))
            with mock.patch.object(fas, "collect_launchd", return_value={}):
                with mock.patch.object(
                    fas,
                    "collect_git",
                    return_value={"branch": "test", "head": "abc", "dirty": False},
                ):
                    with mock.patch.object(fas, "pgrep_first", return_value=None):
                        status = fas.build(root)
        self.assertTrue(status["absent_directive_files"]["all_absent"])
        self.assertEqual(
            set(status["absent_directive_files"]["missing"]),
            set(fas.ABSENT_DIRECTIVE_FILES),
        )


class TestCriticalPathSemantics(unittest.TestCase):
    def test_dag_encodes_capability_and_director_walls(self) -> None:
        dag = fas.build_dag("t")
        pairs = {(e["from"], e["to"]) for e in dag["edges"]}
        self.assertIn(("FA02", "FA05"), pairs)
        self.assertIn(("FA02", "FA06"), pairs)
        self.assertIn(("FA06", "FA10"), pairs)
        self.assertIn(("FA06", "FA12"), pairs)
        self.assertIn(("FA02", "FA04"), pairs)
        walls = " ".join(w["wall"] for w in dag["critical_path"]["hard_walls"])
        self.assertIn("Math-Preserve-v2", walls)
        self.assertIn("Director", walls)
        self.assertIn("HIDE_KERNEL_TURN", walls)


if __name__ == "__main__":
    unittest.main()
