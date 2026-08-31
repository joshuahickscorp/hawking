"""The ceiling audit must CHECK its quotes, not record them.

roof_anchor exists to find numbers promoted across hops without their roof. Its
own CEILING_AUDIT stored quoted_value into a dict and never compared it against
the receipt it named - so when the budget's demonstrated rung moved 47.97 ->
47.25, the audit went on quoting 47.97 with every test green. An audit that
records a number instead of checking it is the defect it exists to find.
"""
from __future__ import annotations

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
    import json

    from tools.future import causal_budget_71 as cb71

    canonical = round(cb71.tps(cb71.token_ms(cb71.CLEAN_GEMV_GB_S)), 2)
    for rel, path in (
        ("receipts/future/CAPABILITY_INFORMATION_MAP.json",
         ["answers", "roof_movement_on_the_71tps_ladder", "quoted_roof_on_todays_bytes"]),
        ("receipts/future/IMPROVEMENT_METABOLISM.json",
         ["cited", "causal_budget", "roof_on_todays_bytes_cited_tps"]),
    ):
        cur = json.loads((ra.REPO / rel).read_text())
        for k in path:
            cur = cur[k]
        assert cur == pytest.approx(canonical, abs=0.01), f"{rel} quotes {cur}, budget says {canonical}"
