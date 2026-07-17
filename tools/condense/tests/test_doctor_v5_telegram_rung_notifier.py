#!/usr/bin/env python3.12
"""Offline tests for the Doctor V5 Telegram rung notifier."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
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
            "cell_identity_sha256": f"{index + 1:064x}"[-64:],
            "adapter_id": "test-adapter",
            "request_sha256": f"{index + 321:064x}"[-64:],
            "result_paths": {},
            "result_sha256": None,
            "disposition_path": "",
            "disposition_sha256": None,
            "exact_stored_parameter_count": 1_000_000,
        })
    campaign = {
        "schema": notifier.CAMPAIGN_SCHEMA,
        "version": "test-v1",
        "plan_sha256": "a" * 64,
        "source_deletion_permitted": False,
        "cells": cells,
        "counts": {"complete": 4, "blocked-execution": 0},
        "queue_status": "running-cell",
    }
    _seal_campaign(campaign)
    return campaign


def _seal_campaign(campaign: dict) -> None:
    campaign.pop("campaign_sha256", None)
    campaign["campaign_sha256"] = notifier._hash_value(campaign)


def _result_document(cell: dict, row: dict | None = None) -> dict:
    row = row or {
        "actual_bpw": 3.9, "target_bpw": 4.0,
        "ppl_delta": 0.02, "capability_delta": 0.0,
    }
    document = {
        "schema": notifier.RESULT_SCHEMA,
        "policy_version": "test-policy",
        "completed_at": "2026-07-16T00:01:00+00:00",
        "request_sha256": cell["request_sha256"],
        "adapter": {"adapter_id": cell["adapter_id"]},
        "status": "complete",
        "output_artifacts": [],
        "metrics": {
            "campaign_cell": {
                field: cell[field]
                for field in (
                    "branch", "cell_id", "cell_identity_sha256",
                    "model_label", "rate_id",
                )
            },
            "physical_accounting": {
                "all_in_model_payload_bpw": row["actual_bpw"],
                "target_physical_bpw": row["target_bpw"],
                "target_met": row["actual_bpw"] <= row["target_bpw"],
            },
            "quality_observation": {
                "status": "provisional_unsealed",
                "ppl": {"relative_delta": row["ppl_delta"]},
                "capability": {"absolute_delta": row["capability_delta"]},
            },
        },
        "evidence_class": "provisional_engineering_evidence",
        "quality_claims_permitted": False,
        "source_deletion_permitted": False,
    }
    document["result_sha256"] = notifier._hash_value(document)
    return document


@contextmanager
def evidence_workspace(campaign: dict,
                       metrics: dict[str, dict] | None = None):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        results = root / "reports/condense/doctor_v5_ultra/results"
        dispositions = root / "reports/condense/doctor_v5_ultra/dispositions"
        evidence_root = root / "reports/condense/doctor_v5_ultra/test_evidence"
        results.mkdir(parents=True)
        dispositions.mkdir(parents=True)
        evidence_root.mkdir(parents=True)
        for cell in campaign["cells"]:
            status = cell.get("status")
            if status not in notifier.TERMINAL_STATUSES:
                continue
            if status == "complete":
                path = results / cell["cell_id"] / "result.json"
                path.parent.mkdir(parents=True)
                result = _result_document(
                    cell, (metrics or {}).get(cell["branch"])
                )
                path.write_text(json.dumps(result), encoding="utf-8")
                cell["result_paths"] = {
                    "result": str(path.relative_to(root)),
                }
                cell["result_sha256"] = result["result_sha256"]
                cell["disposition_sha256"] = None
                continue
            evidence_path = evidence_root / f"{cell['cell_id']}.json"
            evidence_path.write_text('{"evidence":true}\n', encoding="utf-8")
            evidence_bytes = evidence_path.read_bytes()
            disposition = {
                "schema": notifier.DISPOSITION_SCHEMA,
                "version": campaign["version"],
                "plan_sha256": campaign["plan_sha256"],
                "cell_id": cell["cell_id"],
                "cell_identity_sha256": cell["cell_identity_sha256"],
                "status": status,
                "reason_code": "test-disposition",
                "detail": "Evidence-bound terminal test disposition.",
                "evidence_artifacts": [{
                    "role": "evidence-000",
                    "path": str(evidence_path.relative_to(root)),
                    "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                    "bytes": len(evidence_bytes),
                }],
                "recorded_at": "2026-07-16T00:02:00+00:00",
                "quality_claims_permitted": False,
                "source_deletion_permitted": False,
            }
            disposition["disposition_sha256"] = notifier._hash_value(disposition)
            path = dispositions / f"{cell['cell_id']}.json"
            path.write_text(json.dumps(disposition), encoding="utf-8")
            cell["disposition_path"] = str(path.relative_to(root))
            cell["disposition_sha256"] = disposition["disposition_sha256"]
            cell["result_sha256"] = None
        _seal_campaign(campaign)
        with mock.patch.object(notifier, "ROOT", root), \
                mock.patch.object(notifier, "RESULTS", results), \
                mock.patch.object(notifier, "DISPOSITIONS", dispositions):
            yield root


class TelegramRungNotifierTests(unittest.TestCase):
    def test_exact_four_branch_rung_is_detected_once(self) -> None:
        fixture = campaign_fixture()
        with evidence_workspace(fixture):
            rungs = notifier.complete_rungs(fixture)
        self.assertEqual(1, len(rungs))
        self.assertEqual("rung/0.5B/4bpw", rungs[0]["event_id"])

    def test_incomplete_branch_prevents_rung(self) -> None:
        fixture = campaign_fixture()
        fixture["cells"][3]["status"] = "running"
        with evidence_workspace(fixture):
            self.assertEqual([], notifier.complete_rungs(fixture))

    def test_mixed_terminal_rung_is_evidence_validated(self) -> None:
        fixture = campaign_fixture()
        fixture["cells"][1]["status"] = "negative"
        fixture["cells"][2]["status"] = "unsupported"
        fixture["cells"][3]["status"] = "unsupported"
        with evidence_workspace(fixture):
            rungs = notifier.complete_rungs(fixture)
        self.assertEqual(1, len(rungs))
        self.assertEqual({"codec_control"}, set(rungs[0]["metrics"]))
        self.assertEqual(
            {"doctor_static", "doctor_conditional", "doctor_full"},
            set(rungs[0]["dispositions"]),
        )
        self.assertTrue(notifier._is_sha(rungs[0]["evidence_root_sha256"]))

    def test_tampered_result_fails_closed(self) -> None:
        fixture = campaign_fixture()
        with evidence_workspace(fixture) as root:
            path = root / fixture["cells"][0]["result_paths"]["result"]
            result = json.loads(path.read_text())
            result["metrics"]["physical_accounting"]["all_in_model_payload_bpw"] = 0.1
            path.write_text(json.dumps(result))
            with self.assertRaises(notifier.NotifierError):
                notifier.complete_rungs(fixture)

    def test_tampered_disposition_fails_closed(self) -> None:
        fixture = campaign_fixture()
        fixture["cells"][1]["status"] = "unsupported"
        with evidence_workspace(fixture) as root:
            path = root / fixture["cells"][1]["disposition_path"]
            disposition = json.loads(path.read_text())
            disposition["detail"] = "tampered"
            path.write_text(json.dumps(disposition))
            with self.assertRaises(notifier.NotifierError):
                notifier.complete_rungs(fixture)

    def test_tampered_disposition_artifact_fails_closed(self) -> None:
        fixture = campaign_fixture()
        fixture["cells"][1]["status"] = "unsupported"
        with evidence_workspace(fixture) as root:
            disposition_path = root / fixture["cells"][1]["disposition_path"]
            disposition = json.loads(disposition_path.read_text())
            evidence_path = root / disposition["evidence_artifacts"][0]["path"]
            evidence_path.write_text('{"evidence":false}\n')
            with self.assertRaises(notifier.NotifierError):
                notifier.complete_rungs(fixture)

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
        for cell in fixture["cells"][1:4]:
            cell["status"] = "unsupported"
        with evidence_workspace(fixture) as root:
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

    def test_next_rung_skips_mixed_terminal_rung(self) -> None:
        fixture = campaign_fixture()
        for cell in fixture["cells"][1:4]:
            cell["status"] = "unsupported"
        for offset, branch in enumerate(notifier.BRANCHES, start=4):
            fixture["cells"][offset].update({
                "model_label": "0.5B", "rate_id": "3", "branch": branch,
                "status": "pending",
                "exact_stored_parameter_count": 500_000_000,
            })
        observer = {"eta": {"branch_rate_seconds_per_billion": {
            f"{branch}@3": 100.0 for branch in notifier.BRANCHES
        }}}
        estimate = notifier._next_rung(fixture, observer)
        self.assertEqual("0.5B", estimate["model_label"])
        self.assertEqual("3", estimate["rate_id"])
        self.assertEqual(200.0, estimate["remaining_seconds"])

    def test_mixed_format_scopes_result_to_measured_branches(self) -> None:
        fixture = campaign_fixture()
        fixture["cells"][1]["status"] = "negative"
        fixture["cells"][2]["status"] = "unsupported"
        fixture["cells"][3]["status"] = "unsupported"
        with evidence_workspace(fixture):
            rung = notifier.complete_rungs(fixture)[0]
            with mock.patch.object(notifier, "_health", return_value={
                "pressure": 1, "swap_mb": 10.0,
                "disk_free_gb": 200.0, "thermal_green": True,
            }):
                text = notifier.format_rung(rung, fixture)
        self.assertIn("0.5B @ 4 bpw closed", text)
        self.assertIn("Measured result: GOOD", text)
        self.assertIn("Optimization possible from measured evidence", text)
        self.assertIn("Evidence: PROMOTABLE (measured branches only)", text)
        self.assertIn(
            "Coverage: 1 measured | 1 negative disposition | 2 adaptively deferred",
            text,
        )
        self.assertIn("Negative: static", text)
        self.assertIn("Deferred: conditional, full", text)
        self.assertIn("1/320 measured", text)
        self.assertIn("4/320 terminal", text)

    def test_all_disposition_format_is_unavailable(self) -> None:
        fixture = campaign_fixture()
        for cell in fixture["cells"][:4]:
            cell["status"] = "unsupported"
        with evidence_workspace(fixture):
            rung = notifier.complete_rungs(fixture)[0]
            with mock.patch.object(notifier, "_health", return_value={
                "pressure": 1, "swap_mb": 10.0,
                "disk_free_gb": 200.0, "thermal_green": True,
            }):
                text = notifier.format_rung(rung, fixture)
        self.assertIn("Measured result: UNAVAILABLE ⚪", text)
        self.assertIn(
            "Optimization possible from measured evidence: UNAVAILABLE", text
        )
        self.assertIn("Evidence: DISPOSITION ONLY", text)
        self.assertNotIn("Result: BAD", text)
        self.assertIn("0/320 measured", text)
        self.assertIn("4/320 terminal", text)

    def test_simple_format_has_decision_and_eta_block(self) -> None:
        row = {"actual_bpw": 3.9, "target_bpw": 4.0,
               "physical_target_met": True, "ppl_delta": 0.02,
               "capability_delta": 0.0, "wall_seconds": 3600,
               "attempts": 1, "quality_status": "provisional"}
        with mock.patch.object(notifier, "_health", return_value={
            "pressure": 1, "swap_mb": 10.0,
            "disk_free_gb": 200.0, "thermal_green": True,
        }):
            fixture = campaign_fixture()
            rung = {
                "event_id": "rung/0.5B/4bpw",
                "model_label": "0.5B",
                "rate_id": "4",
                "cells": fixture["cells"][:4],
                "metrics": {branch: dict(row) for branch in notifier.BRANCHES},
                "dispositions": {},
                "evidence_root_sha256": "b" * 64,
            }
            text = notifier.format_rung(
                rung, fixture,
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
        self.assertIn("4/320 measured", text)
        self.assertIn("4/320 terminal", text)
        self.assertIn("provisional", text)

    def test_next_rung_falls_back_to_campaign_rates_when_observer_blocked(self) -> None:
        fixture = campaign_fixture()
        for cell in fixture["cells"][:4]:
            cell.update({
                "started_at": "2026-07-16T00:00:00+00:00",
                "completed_at": "2026-07-16T00:10:00+00:00",
                "exact_stored_parameter_count": 1_000_000_000,
            })
        for offset, branch in enumerate(notifier.BRANCHES, start=4):
            fixture["cells"][offset].update({
                "model_label": "0.5B", "rate_id": "3", "branch": branch,
                "status": "pending",
                "exact_stored_parameter_count": 500_000_000,
            })
        observer = {"eta": {
            "confidence": "blocked",
            "reason": "one or more admitted cells are blocked-execution",
        }}
        estimate = notifier._next_rung(fixture, observer)
        self.assertEqual("0.5B", estimate["model_label"])
        self.assertEqual("3", estimate["rate_id"])
        self.assertEqual(1200.0, estimate["remaining_seconds"])

    def test_eta_block_renders_blocked_reason_and_count(self) -> None:
        fixture = campaign_fixture()
        fixture["counts"]["blocked-execution"] = 8
        observer = {"eta": {
            "confidence": "blocked",
            "reason": "one or more admitted cells are blocked-execution",
            "to_120b_boundary": {"point_seconds": None, "point_at": None},
        }}
        text = "\n".join(notifier._eta_block(fixture, observer))
        self.assertIn(
            "Overall sub-120B: blocked, 8 blocked-execution "
            "(one or more admitted cells are blocked-execution)",
            text,
        )
        self.assertNotIn("Overall sub-120B: ETA learning", text)

    def test_eta_block_renders_through_120b_point(self) -> None:
        fixture = campaign_fixture()
        observer = {"eta": {
            "to_120b_boundary": {"point_at": "2099-08-14T08:00:53+00:00"},
            "through_120b": {"point_at": "2099-08-20T08:00:53+00:00"},
        }}
        text = "\n".join(notifier._eta_block(fixture, observer))
        self.assertIn("Overall sub-120B: Aug 14", text)
        self.assertIn("120B: Aug 20", text)
        self.assertIn("remaining, provisional", text)

    def test_eta_block_renders_through_120b_gated_reason(self) -> None:
        fixture = campaign_fixture()
        observer = {"eta": {
            "confidence": "provisional",
            "to_120b_boundary": {"point_at": "2099-08-14T08:00:53+00:00"},
            "through_120b": {
                "point_seconds": None, "point_at": None,
                "reason": "120B campaign is not fully wired/reviewed/executable",
            },
        }}
        lines = notifier._eta_block(fixture, observer)
        self.assertIn(
            "120B: gated (120B campaign is not fully wired/reviewed/executable)",
            lines,
        )

    def test_horizon_lines_marked_planning_projection(self) -> None:
        fixture = campaign_fixture()
        observer = {"eta": {"branch_seconds_per_billion": {"codec_control": 600.0}}}
        lines = notifier._horizon_block(fixture, observer)
        self.assertIn("DeepSeek-V4-Flash 284B: ~47.3h "
                      "(planning projection, scaffold only, not scheduled)", lines)
        self.assertIn("Kimi-K2.6 1.1T: ~7.6d "
                      "(planning projection, scaffold only, not scheduled)", lines)
        self.assertIn("DeepSeek-V4-Pro 1.6T: ~11.1d "
                      "(planning projection, scaffold only, not scheduled)", lines)
        for cell in fixture["cells"][:4]:
            cell.update({
                "started_at": "2026-07-16T00:00:00+00:00",
                "completed_at": "2026-07-16T00:10:00+00:00",
                "exact_stored_parameter_count": 1_000_000_000,
            })
        fallback_lines = notifier._horizon_block(fixture, None)
        self.assertIn("DeepSeek-V4-Flash 284B: ~47.3h "
                      "(planning projection, scaffold only, not scheduled)",
                      fallback_lines)

    def test_horizon_lines_dropped_before_truncation(self) -> None:
        row = {"actual_bpw": 3.9, "target_bpw": 4.0,
               "physical_target_met": True, "ppl_delta": 0.02,
               "capability_delta": 0.0, "wall_seconds": 3600,
               "attempts": 1, "quality_status": "provisional"}
        fixture = campaign_fixture()
        rung = {
            "event_id": "rung/0.5B/4bpw",
            "model_label": "0.5B",
            "rate_id": "4",
            "cells": fixture["cells"][:4],
            "metrics": {branch: dict(row) for branch in notifier.BRANCHES},
            "dispositions": {},
            "evidence_root_sha256": "b" * 64,
        }
        observer = {"eta": {"branch_seconds_per_billion": {"codec_control": 600.0}}}
        with mock.patch.object(notifier, "_health", return_value={
            "pressure": 1, "swap_mb": 10.0,
            "disk_free_gb": 200.0, "thermal_green": True,
        }):
            full = notifier.format_rung(rung, fixture, observer)
            self.assertIn("Post-120B horizon", full)
            self.assertIn("planning projection", full)
            budget = len(full) - 1
            with mock.patch.object(notifier, "MAX_MESSAGE_CHARS", budget):
                trimmed = notifier.format_rung(rung, fixture, observer)
        self.assertNotIn("planning projection", trimmed)
        self.assertLessEqual(len(trimmed), budget)
        self.assertIn("Provisional until the signed physical release gate.", trimmed)


if __name__ == "__main__":
    unittest.main()
