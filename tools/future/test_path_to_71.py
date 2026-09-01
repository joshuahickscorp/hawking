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


def test_everything_on_record_still_does_not_reach_71():
    """The headline that keeps the campaign honest, rebased twice now.

    The 50 TPS ceiling was true against a 27.2896 ms WALL baseline. G131
    promoted three levers into the parent and measured 21.9464 ms GPU, so the
    ladder now composes to 62.9 - it crosses 50 and 60. The INVARIANT was never
    the number 50; it is that nothing ON RECORD reaches the target, and that the
    best path beats the measured baseline it is built from. Pinning the old
    figure would fail the moment the campaign succeeded, which is a floor that
    punishes progress.
    """
    best = max(p.paths(), key=lambda r: r["tps"])
    baseline = 1000.0 / p.TOKEN_MS
    assert best["tps"] < 71.0, best["tps"]
    assert best["tps"] > baseline, (best["tps"], baseline)


def test_a_lever_already_in_the_baseline_is_never_subtracted_again():
    """The double-count LEVER_PROMOTION_GATE exists to refuse. widen_f4 and the
    ba fusion are INSIDE the 21.9464 ms parent; counting them as rungs claimed
    67.86 TPS where the honest figure is 62.9."""
    assert p.PROMOTED_INTO_BASELINE, "the promoted set must not be empty"
    for row in p.paths():
        for cid in row.get("components", []):
            assert cid not in p.PROMOTED_INTO_BASELINE, (
                f"{row['path']} subtracts {cid}, which is already in the baseline"
            )
    composed = [r for r in p.paths() if r["path"] != "PATH_00"]
    skipped = " ".join(s for r in composed for s in r.get("skipped", []))
    assert "ALREADY IN THE BASELINE" in skipped, (
        "the exclusion must be stated in the receipt, not silently applied"
    )


def test_a_path_whose_every_lever_is_promoted_equals_the_baseline():
    """PATH_01 was 'everything QUALIFIED today' and everything qualified has now
    been promoted, so it is the baseline. That is a real result, not a bug."""
    rows = {r["path"]: r for r in p.paths()}
    # PATH_00 carries the raw baseline; composed rows round to 3 decimals.
    assert abs(rows["PATH_01"]["token_ms"] - rows["PATH_00"]["token_ms"]) < 1e-3
    assert rows["PATH_01"]["components"] == []
    assert len(rows["PATH_01"]["skipped"]) == 2


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


def test_an_unknown_flag_refuses_rather_than_writing_nothing(tmp_path):
    """--build printed the new table, exited 0 and wrote NOTHING, because the
    CLI only knew --record. The receipt stayed stale while the terminal showed
    fresh numbers."""
    import subprocess, sys as _s
    r = subprocess.run([_s.executable, str(Path(p.__file__)), "--bogus"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "unknown flag" in (r.stderr + r.stdout)


def test_both_write_verbs_actually_write(tmp_path, monkeypatch):
    import subprocess, sys as _s
    for flag in ("--record", "--build"):
        r = subprocess.run([_s.executable, str(Path(p.__file__)), flag],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "wrote " in r.stdout, f"{flag} printed no write"


def test_the_receipt_on_disk_matches_what_build_computes():
    """The staleness this obligation is about: a receipt that disagrees with its
    own producer is worse than no receipt."""
    import json as _j
    on_disk = _j.loads((p.REPO / "receipts/future/PATH_TO_71.json").read_text())
    fresh = p.build()
    assert on_disk["gap_to_71"]["best_composed_tps"] == \
        fresh["gap_to_71"]["best_composed_tps"]
    assert on_disk["baseline"]["token_ms"] == fresh["baseline"]["token_ms"]
