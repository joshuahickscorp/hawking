"""Research bus: typed messages, receipt-required MEASURED_FACT, epistemic class."""
from __future__ import annotations

import pytest

from lab.lineage.bus import BusRefusal, EpistemicClass, MessageType, ResearchBus, validate_message
from lab.lineage.canon import labeled_sha
from lab.receipts import verify


def _base(**overrides):
    msg = {
        "type": "HYPOTHESIS",
        "sender": "parent",
        "artifact_sha": labeled_sha("artifact/bus"),
        "runtime_sha": labeled_sha("runtime/bus"),
        "epistemic_class": "HYPOTHESIS",
        "receipt_ref": "",
        "affected_subsystem": "weight_addressing",
        "body": {"text": "per-layer codec is unexplored"},
    }
    msg.update(overrides)
    return msg


def test_all_nine_types_exist() -> None:
    assert {t.value for t in MessageType} == {
        "MEASURED_FACT",
        "HYPOTHESIS",
        "NEGATIVE_SCIENCE",
        "MECHANISM",
        "COUNTEREXAMPLE",
        "REQUEST",
        "PATCH_SUMMARY",
        "PROFILE_DELTA",
        "NEXT_BOTTLENECK",
    }


def test_parent_child_a_child_b_exchange() -> None:
    bus = ResearchBus()
    art = labeled_sha("artifact/bus")
    rt = labeled_sha("runtime/bus")
    parent = bus.publish(
        _base(
            sender="parent",
            type="NEXT_BOTTLENECK",
            epistemic_class="HYPOTHESIS",
            body={"NEXT_BOTTLENECK": "weight_addressing"},
        )
    )
    child_a = bus.publish(
        _base(
            sender="child-a",
            type="NEGATIVE_SCIENCE",
            epistemic_class="NEGATIVE",
            receipt_ref="receipts/ascent-2026-08-16/layer42-collapse.json",
            affected_subsystem="attention",
            body={"text": "this codec failed because attention layer 42 collapsed"},
        )
    )
    child_b = bus.publish(
        _base(
            sender="child-b",
            type="PROFILE_DELTA",
            epistemic_class="MEASURED",
            receipt_ref="receipts/ascent-2026-08-16/profile-delta-child-b.json",
            affected_subsystem="deltanet",
            body={"delta_ns": -120_000},
        )
    )
    for row in (parent, child_a, child_b):
        verify(row, label="bus")
        assert row["artifact_sha"] == art
        assert row["runtime_sha"] == rt
        assert row["epistemic_class"]
        assert row["affected_subsystem"]
    assert [m["sender"] for m in bus.messages()] == ["parent", "child-a", "child-b"]
    assert len(bus.messages(type=MessageType.NEGATIVE_SCIENCE)) == 1


def test_measured_fact_without_receipt_refused() -> None:
    bus = ResearchBus()
    with pytest.raises(BusRefusal, match="MEASURED_FACT without a receipt"):
        bus.publish(
            _base(
                type="MEASURED_FACT",
                epistemic_class="MEASURED",
                receipt_ref="",
                body={"ns": 35_227_918},
            )
        )
    with pytest.raises(BusRefusal, match="receipt reference"):
        validate_message(
            _base(
                type="MEASURED_FACT",
                epistemic_class="MEASURED",
            )
        )
    # Missing key entirely is the same refusal.
    raw = _base(type="MEASURED_FACT", epistemic_class="MEASURED")
    raw.pop("receipt_ref")
    with pytest.raises(BusRefusal, match="MEASURED_FACT"):
        bus.publish(raw)
    assert bus.messages() == []


def test_measured_fact_with_receipt_accepted() -> None:
    bus = ResearchBus()
    sealed = bus.publish(
        _base(
            type="MEASURED_FACT",
            epistemic_class="MEASURED",
            receipt_ref="receipts/ascent-2026-08-16/QWEN38_TOKEN_NS_LEDGER.json",
            body={"complete_token_ns": 35_227_918},
        )
    )
    assert sealed["type"] == "MEASURED_FACT"
    assert sealed["receipt_ref"].endswith("QWEN38_TOKEN_NS_LEDGER.json")


def test_negative_science_may_report_failure_not_impossibility() -> None:
    bus = ResearchBus()
    ok = bus.publish(
        _base(
            type="NEGATIVE_SCIENCE",
            epistemic_class="NEGATIVE",
            receipt_ref="receipts/ascent-2026-08-16/layer42-collapse.json",
            body={
                "text": "this codec failed because attention layer 42 collapsed",
                "impossible": False,
            },
        )
    )
    assert ok["epistemic_class"] == EpistemicClass.NEGATIVE.value
    with pytest.raises(BusRefusal, match="impossibility claim refused"):
        bus.publish(
            _base(
                type="NEGATIVE_SCIENCE",
                epistemic_class="NEGATIVE",
                receipt_ref="receipts/ascent-2026-08-16/layer42-collapse.json",
                body={
                    "text": "this codec is impossible",
                    "impossible": True,
                },
            )
        )


def test_impossibility_requires_saturated_resource() -> None:
    bus = ResearchBus()
    with pytest.raises(BusRefusal, match="named_saturated_resource"):
        bus.publish(
            _base(
                type="NEGATIVE_SCIENCE",
                epistemic_class="IMPOSSIBLE",
                receipt_ref="r1",
                body={"establishes_impossibility": True},
            )
        )
    sealed = bus.publish(
        _base(
            type="NEGATIVE_SCIENCE",
            epistemic_class="IMPOSSIBLE",
            receipt_ref="r1",
            body={
                "establishes_impossibility": True,
                "named_saturated_resource": "unified-memory bandwidth at unique-bytes-once decode ceiling",
                "text": "this pack cannot beat the unique-bytes-once ceiling on this genome",
            },
        )
    )
    assert sealed["epistemic_class"] == "IMPOSSIBLE"


def test_type_class_mismatch_refused() -> None:
    with pytest.raises(BusRefusal, match="cannot carry epistemic_class"):
        validate_message(_base(type="HYPOTHESIS", epistemic_class="MEASURED"))
    with pytest.raises(BusRefusal, match="cannot carry"):
        validate_message(_base(type="MEASURED_FACT", epistemic_class="HYPOTHESIS", receipt_ref="r"))


def test_unknown_type_refused() -> None:
    with pytest.raises(BusRefusal, match="unknown message type"):
        validate_message(_base(type="VIBES"))


def test_corrupt_artifact_sha_refused() -> None:
    with pytest.raises(ValueError, match="artifact_sha"):
        validate_message(_base(artifact_sha="not-a-sha"))


def test_generation_aware_message_requires_complete_provenance() -> None:
    bus = ResearchBus()
    strict = _base(
        generation=7,
        repo_head="a" * 40,
        facet="representation",
        invalidation_condition="artifact or runtime identity changes",
        generation_scope="GENERATION_BOUND",
    )
    row = bus.publish_generation_aware(strict)
    assert row["generation"] == 7
    assert row["repo_head"] == "a" * 40
    assert row["facet"] == "representation"
    for missing in (
        "generation",
        "repo_head",
        "facet",
        "invalidation_condition",
        "generation_scope",
    ):
        broken = dict(strict)
        broken.pop(missing)
        with pytest.raises(BusRefusal):
            ResearchBus().publish_generation_aware(broken)


def test_generation_view_marks_bound_science_stale_without_mutating_log() -> None:
    bus = ResearchBus()
    common = {
        "generation": 7,
        "repo_head": "b" * 40,
        "facet": "kernel",
        "invalidation_condition": "runtime changes",
    }
    bound = bus.publish_generation_aware(
        _base(**common, generation_scope="GENERATION_BOUND")
    )
    structural = bus.publish_generation_aware(
        _base(
            **common,
            type="NEGATIVE_SCIENCE",
            epistemic_class="NEGATIVE",
            generation_scope="STRUCTURALLY_TRANSFERABLE",
        )
    )
    view = bus.generation_view(
        generation=8,
        artifact_sha="c" * 64,
        runtime_sha="d" * 64,
    )
    assert [row["continuity_state"] for row in view] == [
        "STALE_PENDING_REVALIDATION",
        "TRANSFERRED_STRUCTURAL",
    ]
    assert "continuity_state" not in bound
    assert "continuity_state" not in structural
