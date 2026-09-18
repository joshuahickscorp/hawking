"""Future evidence producers import canonical status-causality mechanics.

Their verdict guards intentionally remain local: they protect different domain
fields. This test covers only the shared fallback and five-field predicate that
used to be copied into each module.
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest

from tools.verify import status_causality as sc


FUTURE_SPECS = (
    (
        "odyssey2_law_store",
        "record_law_store_causality",
        {
            "schools": {"Flash": {"physical_status": "metadata_only_weights_not_present"}},
            "probe_kind": "stale-probe",
            "claim_kind": "stale-claim",
        },
        "OVERREACHING",
        "object_absence",
        "object_absence",
    ),
    (
        "qualification_pipeline",
        "record_preflight_causality",
        {
            "status": "READY",
            "blocking_defect_count": 0,
            "would_waste_a_protected_window": False,
            "probe_kind": "stale-probe",
            "claim_kind": "stale-claim",
        },
        "UNTESTED",
        None,
        "stale-claim",
    ),
    (
        "contamination",
        "record_contamination_causality",
        {
            "contamination_class": "HEAVY",
            "probe_kind": "stale-probe",
            "claim_kind": "stale-claim",
        },
        "UNTESTED",
        None,
        "stale-claim",
    ),
)

PUBLIC_SPECS = tuple((module, record) for module, record, *_ in FUTURE_SPECS)
OBSERVED_SPECS = tuple((module, record, prototype) for module, record, prototype, *_ in FUTURE_SPECS)


def _protected_values(module_name: str, result: dict[str, object]) -> tuple[object, ...]:
    if module_name == "odyssey2_law_store":
        schools = result["schools"]
        return (schools["Flash"]["physical_status"],)  # type: ignore[index]
    if module_name == "qualification_pipeline":
        return (
            result["blocking_defect_count"],
            result["would_waste_a_protected_window"],
        )
    return (result["contamination_class"],)


def test_future_modules_have_no_local_causality_fallback_or_predicate():
    for module_name, *_ in FUTURE_SPECS:
        path = Path("tools/future") / f"{module_name}.py"
        source = path.read_text()
        assert "def _bind_emit(" not in source
        assert "sc.emit = emit" not in source
        assert "def records_five_fields(" not in source
        assert "FIVE_RECORDED_FIELDS = sc.FIVE_RECORDED_FIELDS" in source
        assert "records_five_fields = sc.records_five_fields" in source


@pytest.mark.parametrize("module_name,record_name", PUBLIC_SPECS)
def test_future_modules_preserve_canonical_aliases(
    module_name: str, record_name: str
):
    module = importlib.import_module(f"tools.future.{module_name}")
    assert getattr(module, record_name)
    assert module.FIVE_RECORDED_FIELDS is sc.FIVE_RECORDED_FIELDS
    assert module.records_five_fields is sc.records_five_fields


def test_importing_future_modules_does_not_replace_emit():
    before = sc.emit
    for module_name, *_ in FUTURE_SPECS:
        importlib.import_module(f"tools.future.{module_name}")
    assert sc.emit is before


@pytest.mark.parametrize("module_name,record_name,prototype", OBSERVED_SPECS)
def test_future_wrappers_stamp_observed_evidence_without_changing_domain_verdict(
    module_name: str,
    record_name: str,
    prototype: dict[str, object],
):
    module = importlib.import_module(f"tools.future.{module_name}")
    result = copy.deepcopy(prototype)
    before = _protected_values(module_name, result)

    record = getattr(module, record_name)(
        result,
        probe_performed="named probe",
        direct_observation={"measured": True},
        interpretation=None,
        probe_kind=sc.PROBE_MEASURED_FLAGS,
        claim_kind=sc.CLAIM_FIELD_VALUE,
    )

    assert record["verdict"] == sc.SUPPORTED
    assert _protected_values(module_name, result) == before
    for field in sc.FIVE_RECORDED_FIELDS:
        assert result[field] == record[field]
    assert result["causality_verdict"] == sc.SUPPORTED
    assert result["probe_kind"] == sc.PROBE_MEASURED_FLAGS
    assert result["claim_kind"] == sc.CLAIM_FIELD_VALUE
    assert sc.records_five_fields(result)


@pytest.mark.parametrize(
    "module_name,record_name,prototype,expected,expected_record_claim,expected_result_claim",
    FUTURE_SPECS,
)
@pytest.mark.parametrize("observation", (None, "", [], {}))
def test_future_wrappers_keep_their_unsupplied_evidence_behavior(
    module_name: str,
    record_name: str,
    prototype: dict[str, object],
    expected: str,
    expected_record_claim: str | None,
    expected_result_claim: str,
    observation: object,
):
    module = importlib.import_module(f"tools.future.{module_name}")
    result = copy.deepcopy(prototype)
    before_protected = _protected_values(module_name, result)
    before_probe = result["probe_kind"]

    record = getattr(module, record_name)(
        result,
        probe_performed="named but unobserved probe",
        direct_observation=observation,
        interpretation=None,
        probe_kind=sc.PROBE_MEASURED_FLAGS,
        claim_kind=sc.CLAIM_FIELD_VALUE,
    )

    assert record["verdict"] == expected
    assert record["direct_observation"] == ""
    assert record["probe_kind"] == ""
    assert record["claim_kind"] == expected_record_claim
    assert _protected_values(module_name, result) == before_protected
    assert result["probe_kind"] == before_probe
    assert result["claim_kind"] == expected_result_claim
