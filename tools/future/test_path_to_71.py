"""The path set must not flatter itself, and refuted components must stay out."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import path_to_71 as p


def test_refuted_components_are_excluded_and_named():
    r = p.compose(["deltanet_q3"])
    assert r["components"] == []
    assert any("REFUTED" in s for s in r["skipped"])


def test_overlapping_levers_are_not_summed():
    """The guard, held against a synthetic pair.

    It used to be held against aux_u8 and group_size_1024, which attacked the
    same 1.07 GB. The capability screen then refuted group_size_1024 outright,
    so that pair no longer exercises the overlap branch - a refuted component is
    skipped for being refuted, before overlap is ever consulted. Binding the
    invariant to whichever components happen to overlap today is how a guard
    quietly stops guarding, so it is bound to a pair this test owns instead.
    """
    saved = dict(p.COMPONENTS)
    try:
        p.COMPONENTS["_probe_a"] = {"gb_saved": 0.5, "evidence": "PROSPECTIVE"}
        p.COMPONENTS["_probe_b"] = {
            "gb_saved": 0.25, "evidence": "PROSPECTIVE", "overlaps": ["_probe_a"],
        }
        both = p.compose(["_probe_a", "_probe_b"])
        assert "_probe_b" not in both["components"]
        assert any("overlaps" in s for s in both["skipped"])
        # compose() rounds to 4 decimals for the receipt; compare at that precision.
        assert both["gb_removed"] == 0.5
    finally:
        p.COMPONENTS.clear()
        p.COMPONENTS.update(saved)


def test_the_refuted_auxiliary_rungs_stay_out_of_every_path():
    """group_size_1024 and _256 failed their capability screen; 2.9 ms was never real."""
    for cid in ("aux_group_size_1024", "aux_group_size_256"):
        assert p.COMPONENTS[cid]["evidence"] == "REFUTED"
        assert p.COMPONENTS[cid]["gb_saved"] == 0.0
    for row in p.paths():
        assert "aux_group_size_1024" not in row["components"]
        assert "aux_group_size_256" not in row["components"]


def test_a_path_is_only_as_strong_as_its_weakest_component():
    rows = {r["path"]: r for r in p.paths()}
    assert rows["PATH_01"]["weakest_evidence"] == "QUALIFIED"
    for pid, row in rows.items():
        if not row["components"]:
            continue
        tiers = [p.COMPONENTS[c]["evidence"] for c in row["components"]]
        assert row["weakest_evidence"] == min(tiers, key=p.EVIDENCE_ORDER.index), pid


def test_an_unranked_tier_raises_instead_of_inheriting_qualified():
    """The bug this replaces: worst was computed by checking for the literal
    string "PROSPECTIVE", so a DIRTY_DIAGNOSTIC component composed into a path
    reported as QUALIFIED."""
    import pytest
    with pytest.raises(p.UnknownEvidenceTier):
        p._weakest(["QUALIFIED", "VIBES"])
    assert p._weakest(["QUALIFIED", "DIRTY_DIAGNOSTIC"]) == "DIRTY_DIAGNOSTIC"
    assert p._weakest(["QUALIFIED", "MEASURED"]) == "MEASURED"


def test_everything_on_record_does_not_reach_50():
    """The headline that keeps the campaign honest.

    The upper bound is the invariant: nothing on record composes to 50 TPS, let
    alone 71. The lower bound is only that the best path beats the measured
    baseline - it used to be a hard 40.0, which was a snapshot of one afternoon's
    ladder rather than a property, and it broke the moment a capability screen
    refuted the two auxiliary rungs it was silently resting on. A floor that
    fails when the science is CORRECTED is a floor that punishes correction.
    """
    best = max(p.paths(), key=lambda r: r["tps"])
    baseline = 1000.0 / p.TOKEN_MS
    assert best["tps"] < 50.0, best["tps"]
    assert best["tps"] > baseline, (best["tps"], baseline)


def test_the_gap_is_stated_as_a_share_of_remaining_gpu():
    g = p.gap_to_71()
    # This pinned > 8.0, which was a fact about the gap on one afternoon. The
    # fold_addqx complete-token A/B then saved 3.9833 ms where the one-layer
    # probe had projected 1.745, and the gap fell to 7.426 - so the test failed
    # BECAUSE the campaign moved. The invariant is that a gap REMAINS: nothing on
    # record reaches 71, and the day that stops being true this assertion should
    # be deleted deliberately, not tripped over.
    assert g["still_to_remove_ms"] > 0.0, "nothing on record may reach 71 silently"
    # The BAND was pinned twice and tripped twice, both times because the campaign
    # moved: 0.35-0.50 held until the baseline rebased onto the measured
    # post-widen_f4 body and the share fell to 0.314. A share is a ratio between
    # two numbers this module already reports, so assert the ARITHMETIC rather
    # than the value - that catches a broken denominator without pinning an
    # afternoon.
    assert 0.0 < g["still_to_remove_share_of_gpu"] < 1.0
    assert g["still_to_remove_ms"] == pytest.approx(
        g["best_composed_token_ms"] - g["target_token_ms"], abs=1e-3
    )
    assert "does not exist yet" in g["verdict"]


def test_no_qualified_component_lacks_parity_evidence():
    for cid, c in p.COMPONENTS.items():
        if c["evidence"] == "QUALIFIED":
            assert c.get("parity"), cid
            assert c.get("source"), cid
