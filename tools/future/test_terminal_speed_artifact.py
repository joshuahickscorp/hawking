"""The terminal artifact must refuse itself until its own measurements land.

G066 asks for exactly one of two receipts and adds the sentence that makes this
module necessary: "Probably impossible" is not an acceptable output; a proof of
the binding limit is. The two failure modes are opposite and both are easy - a
premature UNLOCK declares victory, a premature ROOF declares a limit while three
measurements are outstanding. Those are the same error.
"""
from __future__ import annotations

import pytest

from tools.future import terminal_speed_artifact as tsa


def test_it_refuses_while_any_prerequisite_is_open():
    open_pre = [r for r in tsa.prerequisite_status() if not r["met"]]
    if not open_pre:
        pytest.skip("all prerequisites landed; the refusal path is no longer reachable")
    with pytest.raises(tsa.TerminalArtifactRefused) as exc:
        tsa.build()
    for row in open_pre:
        assert row["id"] in str(exc.value), "a refusal must name what is missing"


def test_every_prerequisite_names_a_receipt_a_field_and_a_reason():
    rows = tsa.prerequisite_status()
    assert rows, "a terminal artifact with no prerequisites cannot refuse anything"
    for row in rows:
        assert row["receipt"].startswith("receipts/future/")
        assert row["field"]
        assert len(row["why"]) > 80, f"{row['id']} has no stated reason"


def test_a_missing_receipt_is_unmet_not_assumed_met():
    assert tsa._resolved("receipts/future/NO_SUCH_RECEIPT.json", ["x"]) is None


def test_a_present_receipt_missing_the_field_is_still_unmet():
    """Existence is not measurement. The field is what makes it one."""
    assert tsa._resolved(
        "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json", ["not_a_field"]
    ) is None
    assert tsa._resolved(
        "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json", ["measured_now", "tps"]
    ) is not None


def test_reached_is_never_claimed_while_the_baseline_is_unknown():
    hit = tsa.reached_71()
    assert hit["target_tps"] == 71.0
    assert hit["target_ms"] == pytest.approx(14.085, abs=1e-3)
    if isinstance(hit["current_body_ms"], str):
        assert hit["reached"] is False


def test_the_verdict_is_never_probably_impossible():
    v = tsa.which_receipt()
    assert v["emit"] in (None, tsa.UNLOCK_NAME, tsa.ROOF_NAME)
    assert "probably" not in v["why"].lower()
    assert "impossible" not in v["why"].lower()


def test_every_prerequisite_names_a_tool_that_can_write_it():
    """Two of the three originally pointed at filenames nobody ever wrote.

    MLP_GRANULARITY_FALSIFIER.json (the receipt is MLP_REGION_FALSIFIER) and
    TOKEN_REGION_TIMESTAMPS.json (the receipt is ORGAN_BANDWIDTH). Both
    measurements had ALREADY LANDED, so those were permanent false blockers on
    completed work - the failure mode this module exists to prevent, inverted.
    A filename can be invented; the module that writes it cannot.
    """
    assert tsa.check_prerequisites_are_writable() == []


def test_an_unwritable_prerequisite_refuses_the_build(monkeypatch):
    fake = dict(tsa.PREREQUISITES[0])
    fake["written_by"] = "tools/future/no_such_module.py"
    monkeypatch.setattr(tsa, "PREREQUISITES", (fake,) + tsa.PREREQUISITES[1:])
    with pytest.raises(tsa.PrerequisiteUnwritable, match="no_such_module"):
        tsa.build()


def test_the_roof_names_all_five_things_s022_asks_for():
    """"Probably impossible" is not an acceptable output; a proof of the binding
    limit is. These five sections are what makes it a proof rather than a shrug."""
    doc = tsa.build()
    for section in (
        "dominant_remaining_costs",
        "irreducible_current_information",
        "best_representation_and_its_evidence",
        "next_hardware_requirement",
        "next_model_body_alternative",
    ):
        assert section in doc, section
        assert doc[section].get("reading"), f"{section} has no reading"


def test_every_section_cites_a_receipt_on_disk():
    doc = tsa.build()
    for section in (
        "dominant_remaining_costs",
        "irreducible_current_information",
        "best_representation_and_its_evidence",
        "next_hardware_requirement",
        "next_model_body_alternative",
    ):
        blk = doc[section]
        rels = blk.get("sources") or [blk["source"]]
        for rel in rels:
            assert (tsa.REPO / rel).exists(), f"{section} cites missing {rel}"


def test_the_dominant_costs_are_measured_on_the_current_body():
    """27.2896 was the pre-promotion WALL figure. The point of this test is that
    the table describes the body that RUNS, so it reads the live absolute rather
    than pinning yesterday's number - the pre-promotion organ table summed to
    26.7013 ms against a 21.9464 ms body and would have failed this on its own
    terms."""
    import json as _j
    d = tsa.dominant_remaining_costs()
    m = _j.loads(
        (tsa.REPO / "receipts/future/SEALED_DEFAULT_ABSOLUTE.json").read_text()
    )["measured"]
    assert d["token_wall_ms"] == pytest.approx(m["wall_ms_per_token"], abs=1e-3)
    assert d["token_gpu_ms"] == pytest.approx(m["gpu_ms_per_token"], abs=1e-3)
    shares = {r["organ"]: r["share_of_gpu"] for r in d["rows"]}
    assert shares["mlp_gate_up"] > shares["deltanet"], "MLP must still lead"
    assert sum(shares.values()) == pytest.approx(1.0, abs=0.05)


def test_the_organ_rows_reconcile_with_the_token_they_decompose():
    """The pre-promotion table was 4.75 ms ABOVE its own baseline. A
    decomposition that does not sum to its total is not a decomposition."""
    d = tsa.dominant_remaining_costs()
    total = sum(float(r["gpu_ms"]) for r in d["rows"])
    assert abs(total - d["token_gpu_ms"]) / d["token_gpu_ms"] < 0.02
    assert "ORGAN_DECOMPOSITION_SEALED" in d["source"]


def test_the_hardware_requirement_does_not_infer_traffic_from_the_catalog():
    h = tsa.next_hardware_requirement()
    assert h["actual_read_bytes_per_token"] == "UNKNOWN"
    assert h["byte_counter_available"] is False


def test_succession_is_not_argued_on_cognition():
    body = tsa.next_model_body_alternative()
    assert "0.6B" in body["reading"]
    assert "not on the decision failures" in body["reading"]
