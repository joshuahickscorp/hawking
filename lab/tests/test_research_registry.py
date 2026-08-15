"""Offline tests for Bible §4 research-registry types and FRANKENSTEIN mapping."""
from __future__ import annotations

import pytest

from lab.operators.research_registry import (
    BFURDSKEffect,
    ResearchItem,
    ResearchRegistry,
    ResearchRegistryError,
    ResearchVerdict,
    item_from_mapping,
    map_frankenstein_verdict,
)


def _item(
    item_id: str = "r1",
    *,
    mechanism: str = "residual_bridge",
    verdict: ResearchVerdict = ResearchVerdict.ADMIT_TO_RUNTIME,
) -> ResearchItem:
    return ResearchItem(
        item_id=item_id,
        mechanism=mechanism,
        hypothesis="reversible residual bridges beat weight merge under H mismatch",
        expected_bfurdsk=BFURDSKEffect(B="neutral", F="small_add", U="neutral", R="high", D="neutral", S="neutral", K="n/a"),
        source_geometry="H_donor=6144,H_host=4096 additive residual",
        prototype="frankenstein_bridges.py V0",
        measured_result="pending V0 seal",
        capability_risk="bounded interference on always-on path",
        gravity_implication="no layout change; adapter archive only",
        runtime_implication="one extra residual matmul per site",
        reopen_condition="if V0 shows secondary regressions, consider multi-adapter hub",
        verdict=verdict,
        citations=("Houlsby 2019", "FRANKENSTEIN_ARCHITECTURE_OPTIONS.md"),
        constraints_checked=("dim_mismatch", "frozen_router", "small_fit", "reversible"),
    )


def test_all_bible_verdicts_exist() -> None:
    names = {v.value for v in ResearchVerdict}
    assert names == {
        "ADMIT_TO_GRAVITY",
        "ADMIT_TO_RUNTIME",
        "ADMIT_TO_KERNEL",
        "DEFER",
        "REJECT",
    }


def test_frankenstein_verdict_mapping() -> None:
    assert map_frankenstein_verdict("RULED OUT") is ResearchVerdict.REJECT
    assert map_frankenstein_verdict("PRIMARY") is ResearchVerdict.ADMIT_TO_RUNTIME
    assert map_frankenstein_verdict("HIGH RISK / DEFER") is ResearchVerdict.DEFER
    assert map_frankenstein_verdict("BEST POST-V0 UPGRADE") is ResearchVerdict.DEFER
    with pytest.raises(ResearchRegistryError):
        map_frankenstein_verdict("totally unknown")


def test_registry_records_and_counts() -> None:
    reg = ResearchRegistry()
    reg.record(_item("a", verdict=ResearchVerdict.ADMIT_TO_RUNTIME))
    reg.record(
        _item(
            "b",
            mechanism="weight_merge_slerp",
            verdict=ResearchVerdict.REJECT,
        )
    )
    snap = reg.as_dict()
    assert snap["item_count"] == 2
    assert snap["counts_by_verdict"]["REJECT"] == 1
    assert snap["counts_by_verdict"]["ADMIT_TO_RUNTIME"] == 1


def test_duplicate_item_id_refused() -> None:
    reg = ResearchRegistry()
    reg.record(_item("dup"))
    with pytest.raises(ResearchRegistryError):
        reg.record(_item("dup"))


def test_research_phase_complete() -> None:
    reg = ResearchRegistry()
    required = ["residual_bridge", "multi_adapter_hub"]
    assert reg.research_phase_complete(required) is False
    reg.record(_item("a", mechanism="residual_bridge"))
    assert reg.incomplete_mechanisms(required) == ["multi_adapter_hub"]
    reg.record(
        _item(
            "b",
            mechanism="multi_adapter_hub",
            verdict=ResearchVerdict.DEFER,
        )
    )
    assert reg.research_phase_complete(required) is True


def test_item_from_mapping_roundtrip() -> None:
    original = _item("rt")
    restored = item_from_mapping(original.as_dict())
    assert restored.item_id == "rt"
    assert restored.verdict is ResearchVerdict.ADMIT_TO_RUNTIME
    assert restored.expected_bfurdsk.B == "neutral"


def test_empty_mechanism_refused() -> None:
    with pytest.raises(ResearchRegistryError):
        ResearchItem(
            item_id="x",
            mechanism=" ",
            hypothesis="h",
            expected_bfurdsk=BFURDSKEffect(),
            source_geometry="g",
            prototype="p",
            measured_result="m",
            capability_risk="c",
            gravity_implication="g",
            runtime_implication="r",
            reopen_condition="never",
            verdict=ResearchVerdict.REJECT,
        )
