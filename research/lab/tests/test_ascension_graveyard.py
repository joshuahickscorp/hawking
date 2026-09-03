"""Offline tests for Bible §32 Ascension Graveyard (ramanujan semantics, no writes)."""
from __future__ import annotations

import pytest

from lab.operators.ascension_graveyard import (
    AscensionGraveyard,
    BIBLE_FAILURE_CLASSES,
    BurialRecord,
    FailureClass,
    GraveyardError,
    ensure_graveyard_checked,
)


def _burial(
    burial_id: str = "g1",
    *,
    mechanism: str = "prompt_independent_kv_reuse",
    failure_class: FailureClass = FailureClass.PROMPT_INDEPENDENT_COLLAPSE,
) -> BurialRecord:
    return BurialRecord(
        burial_id=burial_id,
        mechanism=mechanism,
        model_geometry="any decoder, prompt-dependent activations",
        measured_outcome="capability collapse on held-out prompts",
        failure_reason="reused prompt-dependent activations without proof",
        reopen_condition="mathematical proof of prompt-independence + new measurement",
        failure_class=failure_class,
        citations=("HAWKING_ASCENSION_BIBLE.md §32",),
    )


def test_bible_failure_classes_catalogue() -> None:
    values = {fc.value for fc in BIBLE_FAILURE_CLASSES}
    assert "prompt_independent_collapse" in values
    assert "beats_null_misuse" in values
    assert "median_masking" in values
    assert "unmeasured_gpu_claims" in values
    assert "storage_accumulation" in values
    assert "ignored_eviction" in values
    assert "capability_loss_after_compression" in values
    assert FailureClass.OTHER not in BIBLE_FAILURE_CLASSES


def test_bury_and_check_proposal_refuses_repeat() -> None:
    gy = AscensionGraveyard()
    gy.bury(_burial())
    gate = gy.check_proposal(mechanism="prompt_independent_kv_reuse")
    assert gate["status"] == "REFUSED"
    assert gate["may_proceed"] is False
    assert len(gate["matching_burials"]) == 1


def test_check_proposal_clear_when_unknown() -> None:
    gy = AscensionGraveyard()
    gate = gy.check_proposal(mechanism="brand_new_mechanism")
    assert gate["status"] == "CLEAR"
    assert gate["may_proceed"] is True


def test_new_premise_requires_tribunal_not_auto_clear() -> None:
    gy = AscensionGraveyard()
    gy.bury(_burial())
    gate = gy.check_proposal(
        mechanism="prompt_independent_kv_reuse",
        new_premise="proved prompt-independent for static RoPE tables only",
    )
    assert gate["status"] == "PREMISE_REVIEW_REQUIRED"
    assert gate["may_proceed"] is False  # scaffold never auto-clears


def test_free_resurrection_refused() -> None:
    gy = AscensionGraveyard()
    gy.bury(_burial())
    with pytest.raises(GraveyardError):
        gy.revive("g1", because="I changed my mind", premise_change_evidence="")


def test_revive_with_premise_change() -> None:
    gy = AscensionGraveyard()
    gy.bury(_burial())
    revived = gy.revive(
        "g1",
        because="new independent measurement under revised geometry",
        premise_change_evidence="verifier_event:proof_of_static_table_reuse#42",
    )
    assert revived.buried is False
    assert gy.by_mechanism("prompt_independent_kv_reuse") == []


def test_double_bury_refused() -> None:
    gy = AscensionGraveyard()
    gy.bury(_burial())
    with pytest.raises(GraveyardError):
        gy.bury(_burial())


def test_ensure_graveyard_checked_raises() -> None:
    gy = AscensionGraveyard()
    gy.bury(_burial(mechanism="beats_null_misuse_demo"))
    with pytest.raises(GraveyardError):
        ensure_graveyard_checked(gy, mechanism="beats_null_misuse_demo")


def test_snapshot_declares_ramanujan_kinship() -> None:
    gy = AscensionGraveyard()
    snap = gy.as_dict()
    assert snap["semantics"]["free_resurrection_refused"] is True
    assert snap["semantics"]["ramanujan_stores_write_paths_untouched"] is True
    assert "Stores.bury/revive" in snap["semantics"]["adapts"]
