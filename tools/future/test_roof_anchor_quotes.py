"""The ceiling audit must CHECK its quotes, not record them.

roof_anchor exists to find numbers promoted across hops without their roof. Its
own CEILING_AUDIT used to copy quoted_value into a dict. Checking that copy
against the receipt (QuoteDrift at construction) caught a silent 47.97 -> 47.25,
then PATH_TO_71 was rebased 42.36 -> 49.8 and collection itself raised against
a producer that had moved. An audit that carries a source reference and
resolves at evaluation time follows the producer; a claimed quote that is
actually wrong still raises QuoteDrift.
"""
from __future__ import annotations

import json

import pytest

from tools.future import roof_anchor as ra


def test_every_row_is_checked_or_says_why_not():
    for row in ra.CEILING_AUDIT:
        if row.get("quote_checked"):
            continue
        assert row.get("why_not_checked"), f"{row['id']} silently skipped its quote"


def test_most_rows_actually_resolve():
    checked = [r for r in ra.CEILING_AUDIT if r.get("quote_checked")]
    assert len(checked) >= len(ra.CEILING_AUDIT) - 1


def test_a_wrong_quote_raises():
    with pytest.raises(ra.QuoteDrift):
        ra._audit_row(
            id="deliberate", receipt="receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
            field="ladder[rung=71 TPS].tps", quoted_value=1.0,
            rests_on_roof_id=None, roof_named_in_record=False, defect=None, reading="x",
        )


def test_unresolvable_field_is_recorded_not_passed():
    row = ra._audit_row(
        id="nope", receipt="receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
        field="no.such.path", quoted_value=1.0,
        rests_on_roof_id=None, roof_named_in_record=False, defect=None, reading="x",
    )
    assert row["quote_checked"] is False and row["why_not_checked"]


def test_selector_grammar():
    f = ra._resolve_field
    b = "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"
    assert f(b, "ladder[rung=71 TPS].tps") == 71.0
    assert f(b, "measured_now.active_bytes") == 9_878_901_136
    assert f(b, "ladder[rung=no such rung].tps") is ra._UNRESOLVABLE
    assert f("receipts/future/NO_SUCH.json", "x") is ra._UNRESOLVABLE


def test_inherited_quotes_are_checked_against_their_origin():
    """A quote can match its own receipt and still be stale at the origin.

    Both inheriting receipts READ the figure rather than hard-coding it, so once
    the check fired they were rebuilt and 65.15 propagated. The invariant is that
    inheritance is checked and currently consistent - not the digits, and not the
    fact that two rows happened to be stale on the day this was written.
    """
    inherited = [r for r in ra.CEILING_AUDIT if "inherited_quote_is_stale" in r]
    assert len(inherited) >= 2, "the hops check must still cover the inheriting rows"
    for r in inherited:
        assert r["inherited_quote_is_stale"] is False, (
            f"{r['id']} quotes {r['quoted_value']} while its origin says {r['origin_value']}"
        )
        assert r["quoted_value"] == r["origin_value"]


def test_every_route_to_the_roof_agrees():
    """capability_information_map re-implements the budget's reconstruction twice.

    Both copies summed only the organs plus the host gap, so when the budget grew
    UNATTRIBUTED_GPU_MS they kept quoting a 66.54 roof against the budget's 65.15
    - and because they DERIVE the figure rather than reading it, rebuilding the
    receipt did not fix it. One ceiling reached by three routes must be one number.
    """
    from tools.future import causal_budget_71 as cb71

    canonical = round(cb71.tps(cb71.token_ms(cb71.CLEAN_GEMV_GB_S)), 2)
    for rel, field in (
        ("receipts/future/CAPABILITY_INFORMATION_MAP.json",
         "answers.roof_movement_on_the_71tps_ladder.quoted_roof_on_todays_bytes"),
        ("receipts/future/IMPROVEMENT_METABOLISM.json",
         "cited.causal_budget.roof_on_todays_bytes_cited_tps"),
    ):
        cur = ra._resolve_field(rel, field)
        assert cur is not ra._UNRESOLVABLE
        assert cur == pytest.approx(canonical, abs=0.01), f"{rel} quotes {cur}, budget says {canonical}"


def test_every_audit_row_carries_a_source_reference():
    for row in ra.CEILING_AUDIT:
        src = row.get("source")
        assert src, f"{row['id']} has no source reference"
        assert src["artifact"], row["id"]
        assert src["field"], row["id"]
        assert isinstance(src["tolerance"], float), row["id"]
        assert src["tolerance"] >= 0, row["id"]


def test_source_ref_refuses_a_missing_part():
    with pytest.raises(ra.RoofAnchorError, match="artifact path"):
        ra.source_ref(artifact="", field="x", tolerance=1e-6)
    with pytest.raises(ra.RoofAnchorError, match="field path"):
        ra.source_ref(artifact="receipts/future/PATH_TO_71.json", field="", tolerance=1e-6)
    with pytest.raises(ra.RoofAnchorError, match="tolerance"):
        ra.source_ref(
            artifact="receipts/future/PATH_TO_71.json",
            field="gap_to_71.best_composed_tps",
            tolerance=None,
        )
    with pytest.raises(ra.RoofAnchorError, match="source reference"):
        ra._audit_row(
            id="no_source",
            rests_on_roof_id=None,
            roof_named_in_record=False,
            defect=None,
            reading="x",
        )


def test_mutating_the_source_propagates_to_the_audit(tmp_path):
    """PATH_TO_71 was rebased 42.36 -> 49.8 and a static audit still carried 42.36.

    Collection then raised QuoteDrift against a producer that had moved. An
    audit that carries a source reference must follow the source: mutating
    the producer updates the audit instead of breaking it. That is the
    producer/consumer relationship this lane exists to establish.
    """
    producer = tmp_path / "PATH_TO_71.json"
    producer.write_text(json.dumps({"gap_to_71": {"best_composed_tps": 42.36}}))
    src = ra.source_ref(
        artifact=str(producer),
        field="gap_to_71.best_composed_tps",
        tolerance=1e-6,
    )
    row = ra._audit_row(
        id="path_to_71_best_composed",
        source=src,
        rests_on_roof_id=None,
        roof_named_in_record=False,
        defect=None,
        reading="component composition",
    )
    assert row["quoted_value"] == pytest.approx(42.36)
    assert row["quote_checked"] is True

    producer.write_text(json.dumps({"gap_to_71": {"best_composed_tps": 49.8}}))
    row2 = ra._audit_row(
        id="path_to_71_best_composed",
        source=src,
        rests_on_roof_id=None,
        roof_named_in_record=False,
        defect=None,
        reading="component composition",
    )
    assert row2["quoted_value"] == pytest.approx(49.8)
    assert row2["quote_checked"] is True
    assert row2["source"] == src


def test_a_genuinely_wrong_quote_is_still_caught(tmp_path):
    """Do not fix drift by weakening the guard.

    When a caller claims 42.36 against a source that carries 49.8, that is a
    wrong quote and QuoteDrift must still fire. Following the producer is
    not the same as ignoring disagreement with a claimed number.
    """
    producer = tmp_path / "PATH_TO_71.json"
    producer.write_text(json.dumps({"gap_to_71": {"best_composed_tps": 49.8}}))
    with pytest.raises(ra.QuoteDrift, match="49.8"):
        ra._audit_row(
            id="path_to_71_best_composed",
            source=ra.source_ref(
                artifact=str(producer),
                field="gap_to_71.best_composed_tps",
                tolerance=1e-6,
            ),
            quoted_value=42.36,
            rests_on_roof_id=None,
            roof_named_in_record=False,
            defect=None,
            reading="stale copy",
        )
