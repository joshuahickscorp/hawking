#!/usr/bin/env python3.12
"""Offline tests for the Doctor V5 Telegram rung notifier."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


CONDENSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONDENSE))
import doctor_v5_telegram_rung_notifier as notifier


def campaign_fixture() -> dict:
    cells = []
    for index in range(320):
        # Production sorts model labels numerically; filler cells must keep the
        # same label shape without completing another target rung.
        label = "0.5B" if index < 40 else "1000B"
        rate = "4" if index < 4 else "0.1"
        branch = notifier.BRANCHES[index] if index < 4 else "codec_control"
        cells.append({
            "cell_id": f"cell-{index}", "model_label": label,
            "rate_id": rate, "branch": branch,
            "status": "complete" if index < 4 else "pending",
            "result_sha256": f"{index:064x}"[-64:],
            "exact_stored_parameter_count": 1_000_000,
        })
    return {"cells": cells, "counts": {"complete": 4, "blocked-execution": 0},
            "queue_status": "running-cell"}


class TelegramRungNotifierTests(unittest.TestCase):
    def test_exact_four_branch_rung_is_detected_once(self) -> None:
        rungs = notifier.complete_rungs(campaign_fixture())
        self.assertEqual(1, len(rungs))
        self.assertEqual("rung/0.5B/4bpw", rungs[0]["event_id"])

    def test_incomplete_branch_prevents_rung(self) -> None:
        fixture = campaign_fixture()
        fixture["cells"][3]["status"] = "running"
        self.assertEqual([], notifier.complete_rungs(fixture))

    def test_state_seal_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            state_path = Path(raw) / "state.json"
            state = {"schema": notifier.STATE_SCHEMA, "created_at": "x",
                     "updated_at": "x", "primed": True, "delivered": {},
                     "important_events": {}}
            state["state_sha256"] = notifier._hash_value(state)
            state_path.write_text(json.dumps(state))
            with mock.patch.object(notifier, "STATE", state_path):
                self.assertTrue(notifier._state()["primed"])
                tampered = json.loads(state_path.read_text())
                tampered["primed"] = False
                state_path.write_text(json.dumps(tampered))
                with self.assertRaises(notifier.NotifierError):
                    notifier._state()

    def test_delivery_is_idempotent(self) -> None:
        fixture = campaign_fixture()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state_path, lock_path = root / "state.json", root / "lock"
            state = {"schema": notifier.STATE_SCHEMA, "created_at": "x",
                     "updated_at": "x", "primed": True, "delivered": {},
                     "important_events": {}}
            state["state_sha256"] = notifier._hash_value(state)
            notifier._atomic_json(state_path, state)
            sent: list[str] = []
            def sender(message: str) -> dict:
                sent.append(message); return {"message_id": len(sent), "sent_at": "x"}
            with mock.patch.object(notifier, "STATE", state_path), \
                    mock.patch.object(notifier, "LOCK", lock_path), \
                    mock.patch.object(notifier, "OUTPUT_ROOT", root), \
                    mock.patch.object(notifier, "CAMPAIGN", root / "campaign.json"), \
                    mock.patch.object(notifier, "OBSERVER", root / "missing.json"), \
                    mock.patch.object(notifier, "format_rung", return_value="rung"):
                notifier._atomic_json(root / "campaign.json", fixture)
                first = notifier.run_once(sender=sender)
                second = notifier.run_once(sender=sender)
            self.assertEqual(1, first["sent"])
            self.assertEqual(0, second["sent"])
            self.assertEqual(["rung"], sent)

    def test_keychain_values_never_enter_launch_arguments(self) -> None:
        with mock.patch.object(notifier, "_keychain_get", return_value="configured"), \
                mock.patch.object(notifier.subprocess, "run") as run, \
                tempfile.TemporaryDirectory() as raw:
            run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch.object(notifier, "PLIST", Path(raw) / "agent.plist"), \
                    mock.patch.object(notifier, "OUTPUT_ROOT", Path(raw)):
                notifier.install_launch_agent()
                data = (Path(raw) / "agent.plist").read_bytes()
            self.assertNotIn(b"configured", data)
            self.assertNotIn(notifier.TOKEN_SERVICE.encode(), data)

    def test_decision_is_good_when_one_branch_passes_quality_floor(self) -> None:
        metrics = {
            branch: {"actual_bpw": 4.1, "target_bpw": 4.0,
                     "ppl_delta": 0.90, "capability_delta": 0.0}
            for branch in notifier.BRANCHES
        }
        metrics["doctor_static"] = {
            "actual_bpw": 3.99, "target_bpw": 4.0,
            "ppl_delta": 0.079, "capability_delta": 0.05,
        }
        decision = notifier._rung_decision(metrics)
        self.assertEqual("GOOD", decision["result"])
        self.assertTrue(decision["optimization_possible"])
        self.assertEqual("doctor_static", decision["best_branch"])

    def test_decision_is_bad_and_not_optimizable_when_quality_misses(self) -> None:
        metrics = {
            branch: {"actual_bpw": 4.1, "target_bpw": 4.0,
                     "ppl_delta": 0.081, "capability_delta": 0.0}
            for branch in notifier.BRANCHES
        }
        decision = notifier._rung_decision(metrics)
        self.assertEqual("BAD", decision["result"])
        self.assertFalse(decision["optimization_possible"])
        self.assertEqual("USEFUL NEGATIVE", decision["evidence_value"])
        self.assertEqual(notifier.BRANCHES, tuple(decision["pareto_active"]))

    def test_quality_only_pass_is_bad_but_has_optimization_headroom(self) -> None:
        metrics = {
            branch: {"actual_bpw": 5.3, "target_bpw": 4.0,
                     "ppl_delta": 0.50, "capability_delta": 0.0}
            for branch in notifier.BRANCHES
        }
        metrics["codec_control"].update(actual_bpw=5.081, ppl_delta=0.070)
        metrics["doctor_conditional"].update(actual_bpw=4.964, ppl_delta=0.080)
        decision = notifier._rung_decision(metrics)
        self.assertEqual("BAD", decision["result"])
        self.assertTrue(decision["optimization_possible"])
        self.assertEqual("model + speed", decision["optimization_scope"])
        self.assertEqual("doctor_conditional", decision["best_branch"])
        self.assertIn("+0.964 bpw", decision["reason"])

    def test_7b_result_is_bad_but_exposes_speed_only_headroom(self) -> None:
        metrics = {
            "codec_control": {"actual_bpw": 5.081, "target_bpw": 4.0,
                              "ppl_delta": 0.224, "capability_delta": 0.0},
            "doctor_static": {"actual_bpw": 5.222, "target_bpw": 4.0,
                              "ppl_delta": 0.592, "capability_delta": 0.0},
            "doctor_conditional": {"actual_bpw": 4.964, "target_bpw": 4.0,
                                   "ppl_delta": 0.280, "capability_delta": 0.0},
            "doctor_full": {"actual_bpw": 5.353, "target_bpw": 4.0,
                            "ppl_delta": 0.272, "capability_delta": 0.05},
        }
        decision = notifier._rung_decision(metrics)
        self.assertEqual("BAD", decision["result"])
        self.assertTrue(decision["optimization_possible"])
        self.assertEqual("speed only", decision["optimization_scope"])
        self.assertEqual(["doctor_static"], decision["dominated_branches"])
        self.assertIn("both gates miss", decision["reason"])

    def test_pareto_structure_prunes_only_strictly_dominated_branches(self) -> None:
        metrics = {
            "codec_control": {"actual_bpw": 3.769, "target_bpw": 3.0,
                              "ppl_delta": 0.821, "capability_delta": -0.05},
            "doctor_static": {"actual_bpw": 3.909, "target_bpw": 3.0,
                              "ppl_delta": 1.004, "capability_delta": -0.15},
            "doctor_conditional": {"actual_bpw": 3.652, "target_bpw": 3.0,
                                   "ppl_delta": 1.037, "capability_delta": -0.15},
            "doctor_full": {"actual_bpw": 4.041, "target_bpw": 3.0,
                            "ppl_delta": 1.449, "capability_delta": -0.10},
        }
        decision = notifier._rung_decision(metrics)
        self.assertEqual(
            ["codec_control", "doctor_conditional"], decision["pareto_active"])
        self.assertEqual(
            ["doctor_static", "doctor_full"], decision["dominated_branches"])
        self.assertEqual("speed only", decision["optimization_scope"])

    def test_next_rung_uses_remaining_branch_work(self) -> None:
        fixture = campaign_fixture()
        for offset, branch in enumerate(notifier.BRANCHES, start=4):
            fixture["cells"][offset].update({
                "model_label": "0.5B", "rate_id": "3", "branch": branch,
                "status": "complete" if branch != "doctor_full" else "pending",
                "exact_stored_parameter_count": 500_000_000,
            })
        observer = {"eta": {"branch_rate_seconds_per_billion": {
            "doctor_full@3": 720.0,
        }}}
        estimate = notifier._next_rung(fixture, observer)
        self.assertEqual("0.5B", estimate["model_label"])
        self.assertEqual("3", estimate["rate_id"])
        self.assertEqual(360.0, estimate["remaining_seconds"])

    def test_running_block_precedes_canonical_incomplete_rung(self) -> None:
        fixture = campaign_fixture()
        fixture["cells"][4].update({
            "model_label": "7B", "rate_id": "3", "branch": "codec_control",
            "status": "running", "started_at": None,
            "exact_stored_parameter_count": 7_000_000_000,
        })
        observer = {"eta": {"branch_rate_seconds_per_billion": {
            "codec_control@3": 400.0,
        }}}
        estimate = notifier._next_rung(fixture, observer)
        self.assertEqual("block", estimate["scope"])
        self.assertEqual("7B", estimate["model_label"])
        self.assertEqual(2800.0, estimate["remaining_seconds"])

    def test_simple_format_has_decision_and_eta_block(self) -> None:
        row = {"actual_bpw": 3.9, "target_bpw": 4.0,
               "physical_target_met": True, "ppl_delta": 0.02,
               "capability_delta": 0.0, "wall_seconds": 3600,
               "attempts": 1, "quality_status": "provisional"}
        with mock.patch.object(notifier, "_result_metrics", return_value=row), \
                mock.patch.object(notifier, "_health", return_value={
                    "pressure": 1, "swap_mb": 10.0,
                    "disk_free_gb": 200.0, "thermal_green": True,
                }):
            text = notifier.format_rung(
                notifier.complete_rungs(campaign_fixture())[0], campaign_fixture(),
                {"eta": {"to_120b_boundary": {
                    "point_at": "2099-08-14T08:00:53+00:00",
                }}},
            )
        self.assertIn("Result: GOOD", text)
        self.assertIn("Optimization possible: YES — MODEL", text)
        self.assertIn("Evidence: PROMOTABLE", text)
        self.assertIn("Speed signal: 4/4 branches Pareto-active", text)
        self.assertIn("Reason: physical and quality gates pass", text)
        self.assertIn("ETA\nNext block: none remaining", text)
        self.assertIn("Overall sub-120B: Aug 14", text)
        self.assertIn("provisional", text)


if __name__ == "__main__":
    unittest.main()
