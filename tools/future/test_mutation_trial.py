"""MUTATION TRIAL: the resident must mutate, and read-only analysis must FAIL.

A trial whose every unit is read-only analysis FAILS regardless of event
count. INCONCLUSIVE does not count toward the three. The engine is reached
through orchestration BINDINGS. The HCLI lease is actually taken.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import improvement_trial as it
from tools.future import mutation_engine as me
from tools.future import mutation_trial as mt
from tools.future import orchestration as orch
from tools.future._common import RECEIPTS, REPO, HARDWARE_FIELDS


def test_all_read_only_control_fails_regardless_of_event_count():
    """NEGATIVE CONTROL: the failure the obligation names."""
    empty = mt.negative_control_all_read_only(0)
    fat = mt.negative_control_all_read_only(mt.READ_ONLY_EVENT_COUNT)
    assert empty["must_fail"] is True
    assert fat["must_fail"] is True
    assert empty["verdict"] == "FAIL", empty
    assert fat["verdict"] == "FAIL", fat
    assert empty["n_events"] == 0
    assert fat["n_events"] == mt.READ_ONLY_EVENT_COUNT
    assert fat["n_events"] == 823
    assert empty["n_verified_classes"] == 0
    assert fat["n_verified_classes"] == 0
    blob = " ".join(fat["unmet"])
    assert "read-only analysis" in blob
    assert "regardless of event count" in blob


def test_read_only_agrees_with_improvement_trial_controls():
    row = mt.negative_control_all_read_only()
    assert row["verdict"] == "FAIL"
    assert set(row["agrees_with"]) == set(mt.AGREES_WITH)
    assert "low_payoff_distraction" in row["agrees_with"]
    assert "misleading_narrow_probe" in row["agrees_with"]
    assert set(mt.AGREES_WITH) <= set(it.CONTROL_NAMES)
    low = it.negative_control("low_payoff_distraction")
    narrow = it.negative_control("misleading_narrow_probe")
    assert low["verdict"] == "FAIL", low
    assert narrow["verdict"] == "FAIL", narrow
    assert "low_payoff_distraction" in row["agrees_with_reason"]
    assert "misleading_narrow_probe" in row["agrees_with_reason"]


def test_negative_controls_all_fail_and_harness_breaks_if_one_passes():
    doc = mt.run_negative_controls()
    assert doc["n_fail"] == doc["n_controls"]
    assert doc["n_pass"] == 0
    assert doc["all_failed"] is True
    assert mt.harness_verdict(doc["controls"]) == "OK"
    sabotaged = [dict(c) for c in doc["controls"]]
    sabotaged[0]["verdict"] = "PASS"
    assert mt.harness_verdict(sabotaged) == "BROKEN_HARNESS"


def test_inconclusive_does_not_count_toward_three():
    classes = {
        me.KERNEL_OR_GPU: {
            "mutation_class": me.KERNEL_OR_GPU,
            "proposed": True,
            "applied": True,
            "measured": True,
            "verdict": me.VERDICT_KEPT,
            "before": {"gpu_us_median": 100.0},
            "after": {"gpu_us_median": 80.0},
            "rollback_digest_match": True,
        },
        me.TOKEN_RATE: {
            "mutation_class": me.TOKEN_RATE,
            "proposed": True,
            "applied": True,
            "measured": True,
            "verdict": me.VERDICT_ROLLED_BACK,
            "before": {"gpu_us_median": 100.0},
            "after": {"gpu_us_median": 120.0},
            "rollback_digest_match": True,
        },
        me.REPRESENTATION_BPW: {
            "mutation_class": me.REPRESENTATION_BPW,
            "proposed": True,
            "applied": True,
            "measured": True,
            "verdict": me.VERDICT_INCONCLUSIVE,
            "before": {"gpu_us_median": 100.0},
            "after": {"gpu_us_median": 100.1},
            "rollback_digest_match": True,
        },
    }
    judged = mt.judge(
        {
            "classes": classes,
            "lease": {"lease_calls": 1},
            "driven_through_bindings": True,
            "n_events": 3,
        }
    )
    assert judged["n_verified_classes"] == 2
    assert me.REPRESENTATION_BPW not in judged["verified_classes"]
    assert judged["verdict"] == "FAIL"
    assert any("need 3" in u for u in judged["unmet"])


def test_event_count_cannot_raise_headline():
    judged = mt.judge(
        {
            "classes": {},
            "units": mt.read_only_units(mt.READ_ONLY_EVENT_COUNT),
            "lease": {"lease_calls": 1},
            "driven_through_bindings": True,
            "n_events": mt.READ_ONLY_EVENT_COUNT,
        }
    )
    assert judged["verdict"] == "FAIL"
    assert judged["event_count_cannot_raise_headline"] is True
    assert any("misleading_narrow_probe" in u for u in judged["unmet"])


def test_metal_nil_refuses_rather_than_inconclusive(monkeypatch):
    def boom():
        return {"system_default": None, "n_devices": 0, "devices": []}

    monkeypatch.setattr(mt.mr, "probe", boom)
    with pytest.raises(mt.MetalUnavailable, match="MTLCreateSystemDefaultDevice"):
        mt.require_metal()


def test_mutation_engine_reachable_through_bindings(tmp_path):
    key = "mutation_engine.py"
    assert key in orch.BINDINGS
    me.unbind()
    try:
        engine = orch.resident_mutation_engine(tmp_path)
        proposed = orch.call_bound(key, "propose", "FT.HCLI_SELF.emit-workunits")
        applied = orch.call_bound(key, "apply", proposed)
        assert applied["before_digest"] != applied["after_digest"]
        orch.call_bound(key, "rollback", proposed)
    finally:
        orch.call_bound(key, "unbind")


def test_build_drives_three_verified_classes_with_lease_and_rollback():
    out = mt.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == mt.RECEIPT
    assert doc["schema"] == mt.SCHEMA
    assert doc["seal_sha256"]
    assert doc["verdict"] == "PASS", (doc.get("judgment"), doc.get("class_report"))
    assert doc["took_gpu_lease"] is True
    assert int(doc["lease_calls"]) > 0
    assert doc["lease"]["acquired"] is True
    assert doc["lease"]["flock"] == "LOCK_EX"
    assert doc["driven_through_bindings"] is True
    assert doc["bindings"]["module"] == "mutation_engine.py"
    assert "mutation_engine.py" in orch.BINDINGS
    judged = doc["judgment"]
    assert judged["n_verified_classes"] >= 3
    assert len(judged["verified_classes"]) >= 3
    assert judged["rollback_exercised"] is True
    metal = doc["metal"]
    assert metal["system_default"]
    assert int(metal["n_devices"]) >= 1
    assert "Apple" in str(metal["system_default"])
    # Per-class report: proposed, applied, measured, verdict, rolled back or kept, numbers.
    reports = {r["mutation_class"]: r for r in doc["class_report"]}
    verified = set(judged["verified_classes"])
    for klass in verified:
        row = reports[klass]
        assert row["proposed"] is True
        assert row["applied"] is True
        assert row["measured"] is True
        assert row["verdict"] in (me.VERDICT_KEPT, me.VERDICT_ROLLED_BACK)
        assert row["verdict"] != me.VERDICT_INCONCLUSIVE
        assert row["rollback_digest_match"] is True
        assert row["before"]
        assert row["after"]
        if klass in (me.KERNEL_OR_GPU, me.TOKEN_RATE, me.REPRESENTATION_BPW):
            assert float(row["before"]["gpu_us_median"]) > 0
            assert float(row["after"]["gpu_us_median"]) > 0
            assert row["before"]["gpu_us_median"] != row["after"]["gpu_us_median"]
        if klass == me.PIPELINE_SELF:
            assert row["before"]["units_queued"] == 100
            assert row["after"]["units_queued"] == 25
    harmful = doc["harmful_rollback"]
    assert harmful["verdict"] == me.VERDICT_ROLLED_BACK
    assert harmful["rollback_digest_match"] is True
    # Negative control is in the receipt and FAILed.
    nc = doc["negative_controls"]
    assert nc["all_failed"] is True
    assert set(doc["agrees_with"]) == set(mt.AGREES_WITH)
    for key in HARDWARE_FIELDS:
        if key in doc and isinstance(doc[key], (int, float)):
            raise AssertionError(f"{key} leaked into the trial receipt")
    engine_path = RECEIPTS / mt.ENGINE_RECEIPT
    assert engine_path.is_file()
    engine = json.loads(engine_path.read_text())
    assert engine["schema"] == me.SCHEMA
    assert int(engine["lease_calls"]) > 0
    assert engine["took_gpu_lease"] is True
    assert engine["driven_by"]["bindings"].endswith("mutation_engine.py]") or (
        "mutation_engine.py" in engine["driven_by"]["bindings"]
    )
    assert "mutation_engine.py" in orch.BINDINGS
    assert any("BINDINGS now names mutation_engine.py" in g for g in engine["gaps_closed"])
    assert engine["proofs"]["all_hold"] is True
    # autonomy_run is not edited; the receipt says how the engine is reached.
    assert doc["autonomy_run_not_edited"] is True
    assert "BINDINGS" in doc["autonomy_run_reaches_engine_how"]
    src = Path(REPO / "tools" / "future" / "autonomy_run.py").read_text()
    assert "from tools.future import mutation_engine" not in src
    assert "import mutation_engine" not in src
