"""Hawking gates use one causality-stamping implementation.

The gate reports retain their own status, qualification, and checks.  The
canonical stamper only attaches causality evidence and marks an absent direct
observation as UNTESTED.
"""
from __future__ import annotations

import copy
import importlib
import inspect
from pathlib import Path

import pytest

from tools.verify import status_causality as sc


GATE_SPECS = (
    (
        "hawking.autonomy_gate",
        "hawking/autonomy_gate.py",
        "record_autonomy_causality",
        "hawking/autonomy_gate.py::run_autonomy_gate",
    ),
    (
        "hawking.modellake_gate",
        "hawking/modellake_gate.py",
        "record_modellake_causality",
        "hawking/modellake_gate.py::run_modellake_census",
    ),
    (
        "hawking.native_gate",
        "hawking/native_gate.py",
        "record_native_causality",
        "hawking/native_gate.py::run_native_gate",
    ),
    (
        "hawking.native_mission_gate",
        "hawking/native_mission_gate.py",
        "record_native_mission_causality",
        "hawking/native_mission_gate.py::run_native_mission_gate",
    ),
    (
        "hawking.recovery_gate",
        "hawking/recovery_gate.py",
        "record_recovery_causality",
        "hawking/recovery_gate.py::run_recovery_gate",
    ),
    (
        "hawking.research_gate",
        "hawking/research_gate.py",
        "record_research_causality",
        "hawking/research_gate.py::run_research_gate",
    ),
    (
        "hawking.resident_gate",
        "hawking/resident_gate.py",
        "record_resident_causality",
        "hawking/resident_gate.py::run_resident_gate",
    ),
    (
        "hawking.perception_gate",
        "hawking/perception_gate.py",
        "record_perception_causality",
        "hawking/perception_gate.py::run_perception_gate",
    ),
)


def _report(status: str = "PASSED") -> dict[str, object]:
    return {
        "status": status,
        "qualification": "original qualification",
        "checks": {"one": status == "PASSED"},
        "probe_kind": "stale-probe",
        "claim_kind": "stale-claim",
    }


def test_status_causality_owns_gate_stamper():
    package = importlib.import_module("tools.verify")
    assert inspect.getmodule(sc.stamp_gate) is sc
    assert package.stamp_gate is sc.stamp_gate
    assert sc._calls_emit_time("sc.stamp_gate(report)")


def test_targeted_gates_have_no_local_causality_shims():
    for _, path, _, _ in GATE_SPECS:
        source = Path(path).read_text()
        assert "def _bind_emit(" not in source
        assert "def _record_gate_causality(" not in source
        assert "FIVE_RECORDED_FIELDS = sc.FIVE_RECORDED_FIELDS" in source
        assert "records_five_fields = sc.records_five_fields" in source
        assert "sc.stamp_gate(" in source


def test_importing_targeted_gates_preserves_causality_owner_identity():
    before_emit = sc.emit
    before_stamper = sc.stamp_gate

    for module_name, _, _, _ in GATE_SPECS:
        importlib.import_module(module_name)

    assert sc.emit is before_emit
    assert sc.stamp_gate is before_stamper


@pytest.mark.parametrize("module_name,_,record_name,__", GATE_SPECS)
def test_gate_public_predicate_remains_the_canonical_predicate(
    module_name: str, _: str, record_name: str, __: str
):
    module = importlib.import_module(module_name)
    assert getattr(module, record_name)
    assert module.records_five_fields is sc.records_five_fields


@pytest.mark.parametrize("module_name,_,record_name,source", GATE_SPECS)
def test_gate_writers_match_the_canonical_stamper(
    module_name: str, _: str, record_name: str, source: str
):
    module = importlib.import_module(module_name)
    kwargs = {
        "probe_performed": "named probe",
        "direct_observation": {"measured": True},
        "interpretation": None,
        "probe_kind": sc.PROBE_MEASURED_FLAGS,
        "claim_kind": sc.CLAIM_FIELD_VALUE,
    }
    delegated_report = _report()
    canonical_report = copy.deepcopy(delegated_report)

    delegated = getattr(module, record_name)(delegated_report, **kwargs)
    canonical = sc.stamp_gate(canonical_report, source=source, **kwargs)

    assert delegated == canonical
    assert delegated_report == canonical_report


def test_stamp_gate_preserves_the_gate_verdict_and_stamps_observation():
    report = _report()
    before = copy.deepcopy(report)

    record = sc.stamp_gate(
        report,
        probe_performed="named probe",
        direct_observation={"measured": True},
        probe_kind=sc.PROBE_MEASURED_FLAGS,
        claim_kind=sc.CLAIM_FIELD_VALUE,
        source="test",
    )

    assert record["verdict"] == sc.SUPPORTED
    assert report["status"] == before["status"]
    assert report["qualification"] == before["qualification"]
    assert report["checks"] == before["checks"]
    for field in sc.FIVE_RECORDED_FIELDS:
        assert report[field] == record[field]
    assert report["causality_verdict"] == sc.SUPPORTED
    assert report["probe_kind"] == sc.PROBE_MEASURED_FLAGS
    assert report["claim_kind"] == sc.CLAIM_FIELD_VALUE
    assert sc.records_five_fields(report)


def test_stamp_gate_does_not_use_generic_status_or_falsifier_keywords():
    report = {"id": "only-an-id", "checks": {}}

    record = sc.stamp_gate(
        report,
        probe_performed="named probe",
        direct_observation={"measured": True},
        source="test",
    )

    assert record["status"] == ""
    assert "status" not in report
    with pytest.raises(TypeError):
        sc.stamp_gate(report, status="PASSED")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        sc.stamp_gate(report, falsifier="invented")  # type: ignore[call-arg]


@pytest.mark.parametrize("field", ("status", "qualification", "checks"))
def test_stamp_gate_rejects_an_emit_that_mutates_gate_verdict(
    monkeypatch: pytest.MonkeyPatch, field: str
):
    report = _report()
    original_emit = sc.emit

    def mutating_emit(*args: object, **kwargs: object) -> dict[str, object]:
        if field == "checks":
            report["checks"]["mutated"] = True  # type: ignore[index]
        else:
            report[field] = "mutated"
        return original_emit(*args, **kwargs)

    monkeypatch.setattr(sc, "emit", mutating_emit)
    with pytest.raises(
        RuntimeError, match=r"status_causality\.emit mutated the gate verdict"
    ):
        sc.stamp_gate(
            report,
            probe_performed="named probe",
            direct_observation={"measured": True},
            source="test",
        )


@pytest.mark.parametrize("observation", (None, "", [], {}))
def test_stamp_gate_does_not_invent_an_observation(observation: object):
    report = _report("FAILED")
    before = copy.deepcopy(report)

    record = sc.stamp_gate(
        report,
        probe_performed="named but unobserved probe",
        direct_observation=observation,
        probe_kind=sc.PROBE_MEASURED_FLAGS,
        claim_kind=sc.CLAIM_FIELD_VALUE,
        source="test",
    )

    assert record["verdict"] == sc.UNTESTED
    assert record["direct_observation"] == ""
    assert record["probe_kind"] == ""
    assert record["claim_kind"] is None
    assert report["status"] == before["status"]
    assert report["qualification"] == before["qualification"]
    assert report["checks"] == before["checks"]
    # This is the historical gate contract: absent evidence does not overwrite
    # a pre-existing top-level label with an empty value.
    assert report["probe_kind"] == before["probe_kind"]
    assert report["claim_kind"] == before["claim_kind"]
