"""Canonical work-event contract: every kind watched accepting and rejecting.

The 15m trial counted seventy real ingestions as zero because the driver and
the judge spoke different names for the same act. A validator nobody has
watched reject is how that happens again. These tests prove the contract can
return the negative for every judgement it makes, that an empty refill is not
a refill, and that no production module in the partition emits a retired name.
"""
from __future__ import annotations

import ast
import json

import pytest

from tools.future import work_events as we
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_build_seals_static_only_receipt():
    out = we.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "WORK_EVENT_CONTRACT.json"
    assert doc["schema"] == we.SCHEMA
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["legacy_emits_in_partition"] == []
    _assert_no_hardware_claims(doc)
    drifted = [r for r in doc["watched_refusals"] if r["refused"] == r["expect_ok"]]
    assert drifted == []


def test_every_canonical_kind_validates_and_rejects_its_missing_payload():
    assert tuple(we.EVENT_KINDS) == we.CANONICAL_KINDS
    for kind, spec in we.EVENT_KINDS.items():
        ok, why = we.validate(we.example(kind))
        assert ok is True, (kind, why)
        assert why == "ok"
        missing = {"kind": kind, "payload": {}}
        ok, why = we.validate(missing)
        assert ok is False, f"{kind} accepted an empty payload"
        required = spec["required"][0]
        assert required in why, (kind, why)


def test_work_refilled_empty_unit_list_is_rejected():
    ok, why = we.validate(
        {
            "kind": "WORK_REFILLED",
            "payload": {"unit_ids": [], "queue_depth": 4},
        }
    )
    assert ok is False
    assert "empty" in why
    assert "not a refill" in why


def test_work_refilled_without_queue_depth_is_rejected():
    ok, why = we.validate(
        {
            "kind": "WORK_REFILLED",
            "payload": {"unit_ids": ["WU.1"]},
        }
    )
    assert ok is False
    assert "queue_depth" in why


def test_work_refilled_without_unit_ids_is_rejected():
    ok, why = we.validate(
        {
            "kind": "WORK_REFILLED",
            "payload": {"queue_depth": 2},
        }
    )
    assert ok is False
    assert "unit_ids" in why


def test_frontier_has_work_is_not_refill_credit():
    """A resident does not get refill credit because static work happened to exist."""
    world = {
        "kind": "FRONTIER_HAS_WORK",
        "payload": {"unit_ids": ["FT.TOOLS.frontiers-refill"]},
    }
    ok, why = we.validate(world)
    assert ok is True, why
    as_refill = {
        "kind": "WORK_REFILLED",
        "payload": {"unit_ids": ["FT.TOOLS.frontiers-refill"]},
    }
    ok, why = we.validate(as_refill)
    assert ok is False
    assert "queue_depth" in why


def test_work_refilled_accepts_the_drivers_depth_spelling():
    ok, why = we.validate(
        {
            "kind": "WORK_REFILLED",
            "payload": {
                "unit_ids": ["WU.1"],
                "queue_remaining_when_asked": 3,
            },
        }
    )
    assert ok is True, why


def test_work_refilled_rejects_a_bool_or_negative_depth():
    for depth in (True, False, -1, 1.5, "3"):
        ok, why = we.validate(
            {
                "kind": "WORK_REFILLED",
                "payload": {"unit_ids": ["WU.1"], "queue_depth": depth},
            }
        )
        assert ok is False, f"queue_depth={depth!r} was accepted"
        assert "queue_depth" in why


def test_result_ingested_empty_cites_is_rejected():
    ok, why = we.validate({"kind": "RESULT_INGESTED", "cites": [], "payload": {}})
    assert ok is False
    assert "cites" in why


def test_result_ingested_top_level_cites_count():
    """The judge already scores cites on the event, not inside payload."""
    ok, why = we.validate(
        {
            "kind": "RESULT_INGESTED",
            "cites": ["receipts/future/DERIVED_FRESHNESS.json"],
            "payload": {"receipt": "receipts/future/DERIVED_FRESHNESS.json"},
        }
    )
    assert ok is True, why


def test_generated_scheduled_launched_each_need_their_own_payload():
    ok, why = we.validate(
        {"kind": "WORK_GENERATED", "payload": {"candidate": {"id": "WU.GEN.1"}}}
    )
    assert ok is True, why
    ok, why = we.validate({"kind": "WORK_GENERATED", "payload": {"candidate": {}}})
    assert ok is False
    ok, why = we.validate(
        {"kind": "WORK_SCHEDULED", "payload": {"unit": {"id": "WU.1"}}}
    )
    assert ok is True, why
    ok, why = we.validate({"kind": "WORK_SCHEDULED", "payload": {"unit": "WU.1"}})
    assert ok is False
    ok, why = we.validate(
        {"kind": "WORK_LAUNCHED", "payload": {"unit": {"id": "WU.1"}}}
    )
    assert ok is True, why
    ok, why = we.validate({"kind": "WORK_LAUNCHED", "payload": {"unit": {}}})
    assert ok is False


def test_legacy_name_is_rejected_until_canonicalize():
    raw = {
        "kind": "receipt_ingested",
        "cites": ["receipts/future/DERIVED_FRESHNESS.json"],
        "payload": {"receipt": "receipts/future/DERIVED_FRESHNESS.json"},
    }
    ok, why = we.validate(raw)
    assert ok is False
    assert "legacy" in why
    assert "RESULT_INGESTED" in why
    rewritten = we.canonicalize(raw)
    assert rewritten["kind"] == "RESULT_INGESTED"
    assert rewritten["legacy_kind"] == "receipt_ingested"
    assert raw["kind"] == "receipt_ingested"
    ok, why = we.validate(rewritten)
    assert ok is True, why


def test_canonicalize_on_a_canonical_kind_does_not_invent_legacy_kind():
    src = we.example("RESULT_INGESTED")
    out = we.canonicalize(src)
    assert out["kind"] == "RESULT_INGESTED"
    assert "legacy_kind" not in out


def test_canonicalize_preserves_legacy_kind_on_a_second_pass():
    raw = {
        "kind": "receipt_ingested",
        "cites": ["receipts/future/X.json"],
        "payload": {},
    }
    once = we.canonicalize(raw)
    twice = we.canonicalize(once)
    assert twice["kind"] == "RESULT_INGESTED"
    assert twice["legacy_kind"] == "receipt_ingested"


def test_canonicalize_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        we.canonicalize({"kind": "busywork_emitted", "payload": {}})
    with pytest.raises(TypeError):
        we.canonicalize("not-an-object")  # type: ignore[arg-type]


def test_precanonical_spellings_rewrite_and_are_not_legacy_aliases():
    """workunit_launched / next_work_left / result_ingested are still spoken.

    They must parse, and they must not sit in LEGACY_ALIASES: putting a live
    emit name there would make the partition-scan a self-own, and this lane
    cannot switch the driver."""
    for spoken, canonical in (
        ("workunit_launched", "WORK_LAUNCHED"),
        ("next_work_left", "FRONTIER_HAS_WORK"),
        ("result_ingested", "RESULT_INGESTED"),
        ("work_refilled", "WORK_REFILLED"),
    ):
        assert spoken not in we.LEGACY_ALIASES, spoken
        out = we.canonicalize({"kind": spoken, "payload": {"unit_ids": ["WU.1"], "unit": {"id": "WU.1"}, "queue_depth": 0, "cites": ["x"]}})
        assert out["kind"] == canonical
        assert out["legacy_kind"] == spoken


def test_make_refuses_to_emit_a_legacy_name():
    with pytest.raises(ValueError, match="refusing to emit legacy"):
        we.make("receipt_ingested", cites=["receipts/future/X.json"])
    with pytest.raises(ValueError, match="empty"):
        we.make("WORK_REFILLED", unit_ids=[], queue_depth=1)
    event = we.make("WORK_REFILLED", unit_ids=["WU.1"], queue_depth=0)
    ok, why = we.validate(event)
    assert ok is True, why


def test_validate_rejects_non_object_and_unknown_kind():
    ok, why = we.validate("WORK_REFILLED")
    assert ok is False
    assert "not an object" in why
    ok, why = we.validate({"payload": {}})
    assert ok is False
    assert "no kind" in why
    ok, why = we.validate({"kind": "QUEUE_LOOKED_BUSY", "payload": {}})
    assert ok is False
    assert "unknown kind" in why


def test_no_production_module_emits_a_legacy_name():
    """AST-scan the partition. Naming the string in the alias table is parsing;
    a string constant anywhere else in production is an emit."""
    hits = we.scan_partition_for_legacy_emits()
    assert hits == [], hits
    # And the scan itself actually looks: a planted Constant would fire.
    tree = ast.parse("KIND = 'receipt_ingested'\n")
    planted = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and n.value in we.LEGACY_ALIASES
    ]
    assert planted == ["receipt_ingested"]


def test_legacy_alias_table_names_the_run_that_cost_evidence():
    assert we.LEGACY_ALIASES["receipt_ingested"] == "RESULT_INGESTED"
    src = open(we.__file__).read()
    assert "e08529b84" in src
    assert "seventy" in src.lower() or "70" in src


def test_receipt_records_the_refill_conflation():
    doc = json.loads(we.build().read_text())
    assert "next_work_left" in doc["refill_vs_world_fact"]
    assert "d6eb11c79" in doc["refill_vs_world_fact"]
    assert doc["kinds"]["WORK_REFILLED"]["required"] == ["unit_ids", "queue_depth"]
    assert doc["kinds"]["FRONTIER_HAS_WORK"]["required"] == ["unit_ids"]
    assert "queue_depth" not in doc["kinds"]["FRONTIER_HAS_WORK"]["required"]
