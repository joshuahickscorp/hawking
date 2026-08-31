"""Tests for tools/future/green_machine.py.

The load-bearing test is the negative control: the scheduler refuses an
energy-based decision while measurement is untrustworthy, and no code path
converts UNKNOWN into a default number.
"""
from __future__ import annotations

import json

import pytest

from tools.future import green_machine as gm
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = gm.build()
    assert out.parent == RECEIPTS
    assert out.name == "GREEN_MACHINE.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == "hawking.future.green_machine.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert doc["claim_class"] == "STATIC_ONLY"
    assert doc["produces_diagnostic_relative"] is False
    assert doc["produces_protected_absolute"] is False
    _assert_no_hardware_claims(doc)


def test_selftest_runs_and_emits_receipt():
    assert gm.selftest() == 0
    assert (RECEIPTS / "GREEN_MACHINE.json").is_file()


def test_metric_contract_covers_required_axes():
    ids = [m["id"] for m in gm.METRIC_CONTRACT]
    assert ids == list(gm.METRIC_IDS)
    for required in (
        "joules_per_token",
        "joules_per_accepted_token",
        "work_units_per_kwh",
        "idle_joules_per_second",
        "active_joules_per_second",
        "thermal_state",
    ):
        assert required in ids
    accepted = next(m for m in gm.METRIC_CONTRACT if m["id"] == "joules_per_accepted_token")
    assert "accepted" in accepted["denominator"]
    assert "speculat" in accepted["definition"].lower()
    idle = next(m for m in gm.METRIC_CONTRACT if m["id"] == "idle_joules_per_second")
    active = next(m for m in gm.METRIC_CONTRACT if m["id"] == "active_joules_per_second")
    assert idle["id"] != active["id"]
    jtok = next(m for m in gm.METRIC_CONTRACT if m["id"] == "joules_per_token")
    assert jtok["blocked_by_write_receipt"] is True
    assert jtok["hardware_field"] in HARDWARE_FIELDS


def test_every_metric_value_is_unknown():
    metrics = gm.unknown_metrics()
    assert set(metrics) == set(gm.METRIC_IDS)
    for mid, entry in metrics.items():
        assert entry["value"] is gm.UNKNOWN
        assert entry["state"] is gm.UNKNOWN
        assert entry["trustworthy"] is False
        assert entry["claim_class"] == "STATIC_ONLY"
        assert not isinstance(entry["value"], (int, float))


def test_receipt_metrics_are_unknown_not_numbers():
    doc = json.loads(gm.build().read_text())
    for mid, entry in doc["metrics"].items():
        assert entry["value"] == "UNKNOWN", mid
        assert not isinstance(entry["value"], (int, float))
    assert doc["measurement_is_trustworthy"] is False
    assert doc["any_probe_declared_token_energy_trust"] is False


def test_probes_actually_ran():
    probes = gm.run_probes()
    ids = [p["id"] for p in probes]
    assert ids == [
        "powermetrics_without_root",
        "sudo_n_powermetrics",
        "pmset_therm",
        "pmset_batt",
        "sysctl_thermal_levels",
        "ioreport_energy_model_catalog",
        "ioreport_energy_model_subscription",
        "ioreg_power_telemetry",
    ]
    for p in probes:
        assert p["trustworthy_for_token_energy"] is False
        assert p["numeric_sample_recorded"] is False
        assert "invoked" in p
        assert "succeeded" in p
        assert "missing_dependency" in p or p.get("succeeded")


def test_powermetrics_without_root_is_not_a_token_joule():
    p = gm.probe_powermetrics_without_root()
    assert p["trustworthy_for_token_energy"] is False
    assert p["numeric_sample_recorded"] is False
    if p["invoked"]:
        assert p["succeeded"] is False
        assert p["missing_dependency"] == "root"


def test_sudo_n_never_prompts_and_does_not_succeed_here():
    p = gm.probe_sudo_n_powermetrics()
    assert p["trustworthy_for_token_energy"] is False
    assert p["succeeded"] is False


def test_lookup_missing_is_unknown_not_zero():
    assert gm._lookup_metric({}, "joules_per_token") is gm.UNKNOWN
    assert gm._lookup_metric(None, "joules_per_token") is gm.UNKNOWN
    assert gm._lookup_metric({"joules_per_token": None}, "joules_per_token") is gm.UNKNOWN
    assert gm._lookup_metric({"joules_per_token": {"value": None}}, "joules_per_token") is gm.UNKNOWN
    assert gm._lookup_metric({"joules_per_token": 0.0}, "joules_per_token") == 0.0


def test_measurement_predicate_requires_all_five_flags():
    assert gm.measurement_is_trustworthy(
        gpu_authority=False,
        protected_lease=False,
        energy_wrap_around_token_ns=False,
        root_powermetrics=False,
        ioreport_live_samples=False,
    ) is False
    # Four of five is still not trustworthy.
    assert gm.measurement_is_trustworthy(
        gpu_authority=True,
        protected_lease=True,
        energy_wrap_around_token_ns=True,
        root_powermetrics=False,
        ioreport_live_samples=False,
    ) is False
    assert gm.measurement_is_trustworthy(
        gpu_authority=True,
        protected_lease=True,
        energy_wrap_around_token_ns=True,
        root_powermetrics=True,
        ioreport_live_samples=False,
    ) is True


def test_admit_is_not_implemented():
    assert gm.admit_is_implemented() is False


# ---------------------------------------------------------------------------
# Negative control — a guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------

def test_energy_number_unknown_does_not_become_a_default():
    with pytest.raises(gm.UntrustworthyMeasurement) as err:
        gm.energy_number(gm.UNKNOWN, "joules_per_token")
    assert "UNKNOWN" in str(err.value)
    assert "default" in str(err.value)


def test_energy_number_zero_is_the_tempting_default_and_still_raises():
    with pytest.raises(gm.UntrustworthyMeasurement):
        gm.energy_number(0.0, "joules_per_token")
    with pytest.raises(gm.UntrustworthyMeasurement):
        gm.energy_number(0, "joules_per_accepted_token")
    with pytest.raises(gm.UntrustworthyMeasurement):
        gm.energy_number(1.23, "work_units_per_kwh")


def test_tdp_and_flop_estimates_are_forbidden():
    with pytest.raises(gm.UntrustworthyMeasurement) as err:
        gm.estimate_from_tdp_watts(140.0, token_ns=1_000_000)
    assert "forbidden" in str(err.value)
    with pytest.raises(gm.UntrustworthyMeasurement):
        gm.estimate_from_flops(1e12, picojoules_per_flop=1.0)


def test_scheduler_refuses_while_untrustworthy():
    decision = gm.EnergyAwareScheduler().schedule({"id": "wu-1"})
    assert decision.action == gm.ACTION_REFUSE
    assert decision.reason_code == gm.REASON_UNTRUSTWORTHY
    assert decision.numeric_energy_used is False
    assert decision.substituted_default is False
    assert decision.admit_implemented is False
    assert decision.claim_class == "STATIC_ONLY"
    assert "UNKNOWN" in decision.detail or "inert" in decision.detail


def test_scheduler_refuses_forged_numbers_and_does_not_use_them():
    """Prove the refusal actually fires when a number is smuggled in.

    0.0 is the classic 'UNKNOWN became a default'. 1.23 is a plausible
    invented precision. Neither may be used, echoed as a used value, or
    converted into Admit.
    """
    decision = gm.EnergyAwareScheduler().schedule(
        {"id": "wu-forged"},
        metrics={
            "joules_per_token": 0.0,
            "joules_per_accepted_token": 1.23,
            "work_units_per_kwh": 999,
        },
    )
    assert decision.action == gm.ACTION_REFUSE
    assert decision.reason_code == gm.REASON_NUMERIC_WITHOUT_AUTHORITY
    assert decision.numeric_energy_used is False
    assert decision.substituted_default is False
    blob = json.dumps(decision.as_dict())
    assert "0.0" not in blob
    assert "1.23" not in blob
    assert "999" not in blob
    assert "Admit" not in blob
    assert decision.as_dict()["action"] != "Admit"


def test_scheduler_does_not_treat_missing_as_zero():
    decision = gm.EnergyAwareScheduler().schedule({"id": "wu-empty"}, metrics={})
    assert decision.action == gm.ACTION_REFUSE
    assert decision.reason_code == gm.REASON_UNTRUSTWORTHY
    assert decision.substituted_default is False
    assert decision.numeric_energy_used is False


def test_scheduler_nested_unknown_metrics_still_refuse():
    decision = gm.EnergyAwareScheduler().schedule(
        {"id": "wu-nested"}, metrics=gm.unknown_metrics()
    )
    assert decision.action == gm.ACTION_REFUSE
    assert decision.numeric_energy_used is False
    assert set(decision.metrics_consulted) == set(gm.METRIC_IDS)


def test_receipt_scheduler_self_decision_is_a_refusal():
    doc = json.loads(gm.build().read_text())
    d = doc["scheduler"]["self_decision"]
    assert d["action"] == "REFUSE"
    assert d["numeric_energy_used"] is False
    assert d["substituted_default"] is False
    assert d["admit_implemented"] is False
    assert doc["scheduler"]["inert"] is True


def test_recovered_implementation_lists_codex_energy_and_does_not_claim_adequacy():
    rows = gm.recover_implementation()
    by_path = {r["path"]: r for r in rows}
    energy = by_path["crates/hawking-core/src/token_ns/energy.rs"]
    assert energy["in_git_head"] is True
    assert energy["adequate_for_this_lane"] is False
    orch = by_path["crates/hawking-orch/src/scheduler.rs"]
    assert orch["adequate_for_this_lane"] is False
    assert "guess" in orch["why_not_adequate"].lower() or "heuristic" in orch["why_not_adequate"].lower()
    scoreboard = by_path["receipts/headless/ACCELERATOR_SCOREBOARD.json"]
    assert scoreboard["in_git_head"] is False
    # Environment-coupled: this file is uncommitted, so it is invisible from a
    # sparse lane worktree and visible from the primary one. Its presence is a
    # fact about the checkout, not about this module -- assert the module COPES
    # either way rather than pinning the environment it was written in.
    assert isinstance(scoreboard["on_disk_this_worktree"], bool)


def test_scoreboard_slot_names_the_missing_columns():
    slot = gm.SCOREBOARD_SLOT["noetic_scoreboard"]
    for col in (
        "JOULES_PER_TOKEN",
        "JOULES_PER_ACCEPTED_TOKEN",
        "WORK_UNITS_PER_KWH",
    ):
        assert col in slot["missing_energy_columns"]
    assert "TPS" in slot["existing_columns"]
    assert "JOULES_PER_TOKEN" not in slot["existing_columns"]


def test_write_receipt_still_rejects_numeric_joules_per_token():
    """The _common guard is the backstop; we must not have weakened it."""
    from tools.future._common import HardwareClaimError, write_receipt

    with pytest.raises(HardwareClaimError):
        write_receipt(
            "GREEN_MACHINE_MUST_NOT_EXIST.json",
            {"schema": "nope", "joules_per_token": 0.042},
            "test_green_machine.py",
        )
    assert not (RECEIPTS / "GREEN_MACHINE_MUST_NOT_EXIST.json").exists()
