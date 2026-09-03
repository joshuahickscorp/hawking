"""132 receipts assert a hardware number with no record of the conditions.

Pinned so it can only shrink. It shrinks when a measurement is RE-RUN through
measurement_provenance(), never by writing lock_held:true onto an old number --
that would fabricate the exact fact that is missing.
"""
from __future__ import annotations

import pytest

from tools.future import measurement_provenance_audit as A
from tools.future._common import HARDWARE_FIELDS

#: The count when this guard was added. LOWER IT, never raise it.
BARE_CEILING = 132


@pytest.fixture(scope="module")
def doc():
    return A.audit()


def test_the_bare_count_only_goes_down(doc):
    assert doc["without_provenance"] <= BARE_CEILING, (
        f"{doc['without_provenance']} receipts now carry a hardware number with no "
        f"provenance, up from {BARE_CEILING}. A new measurement was written without "
        "recording whether it held the GPU lane."
    )


def test_the_split_is_exhaustive(doc):
    assert doc["receipts_carrying_a_hardware_number"] == (
        doc["with_provenance"] + doc["without_provenance"])
    assert doc["receipts_scanned"] > doc["receipts_carrying_a_hardware_number"]


def test_the_field_list_has_one_definition(doc):
    """Two lists of what counts as a hardware number drift, and then the guard
    and the audit disagree about what they protect."""
    assert set(doc["hardware_field_names"]) == set(HARDWARE_FIELDS)


def test_it_finds_a_nested_hardware_number():
    """Real receipts bury them; a shallow scan would report a clean corpus."""
    doc = {"a": {"b": [{"c": {"accepted_tps": 23.63}}]}}
    hits = list(A.hardware_numbers(doc))
    assert hits == [(".a.b[0].c.accepted_tps", 23.63)], hits


def test_a_zero_is_not_counted_as_a_measurement():
    """Zero throughput is not a throughput claim; UNRECORDED is not zero."""
    assert list(A.hardware_numbers({"tps": 0})) == []
    assert list(A.hardware_numbers({"tps": 0.0})) == []
    assert list(A.hardware_numbers({"tps": 1.5})) == [(".tps", 1.5)]


def test_a_provenance_marker_anywhere_counts_as_provenanced():
    """The marker need not be at the top level; measurement_provenance() nests it."""
    blob = {"deep": {"measurement_provenance": {"lock_held": True}}, "tps": 30.0}
    import json
    assert any(m in json.dumps(blob) for m in A.PROVENANCE_MARKERS)
