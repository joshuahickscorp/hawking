"""CCL pins. FRONT B (G044). The validator exists to stop a ledger with holes from
reading as a ledger with coverage."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/accelerator"))
import ccl  # noqa: E402


def good(**over):
    base = dict(capability_id="MEMORY.t", cuda_mechanism="m", why_it_exists="w",
                underlying_problem="u", apple_equivalent="a", hawking_equivalent="h",
                semantic_gap="PARTIAL", performance_gap=ccl.unmeasured("none"),
                priority="P1", test_corpus="t", current_winner="w",
                remaining_limitation="l")
    base.update(over)
    return base


def test_every_named_field_is_required():
    for f in ccl.FIELDS:
        d = good()
        d.pop(f)
        with pytest.raises(ccl.CCLError, match="missing"):
            ccl.entry(**d)


def test_parity_claim_without_evidence_is_refused():
    with pytest.raises(ccl.CCLError, match="PARITY CLAIM"):
        ccl.entry(**good(semantic_gap="NONE"))


def test_parity_claim_with_evidence_is_allowed():
    e = ccl.entry(**good(semantic_gap="NONE", evidence="receipts/x.json"))
    assert e["semantic_gap"] == "NONE"


def test_measured_performance_gap_requires_a_receipt():
    with pytest.raises(ccl.CCLError, match="needs a receipt"):
        ccl.entry(**good(performance_gap={"measured": True, "value": "2x"}))


def test_capability_id_must_name_a_real_class():
    with pytest.raises(ccl.CCLError, match="must start with"):
        ccl.entry(**good(capability_id="VIBES.t"))


def test_unknown_gap_and_priority_are_refused():
    with pytest.raises(ccl.CCLError):
        ccl.entry(**good(semantic_gap="SMALLISH"))
    with pytest.raises(ccl.CCLError):
        ccl.entry(**good(priority="URGENT"))


def test_duplicate_ids_refused():
    with pytest.raises(ccl.CCLError, match="duplicate"):
        ccl.build([ccl.entry(**good()), ccl.entry(**good())])


def test_built_ledger_never_claims_parity():
    led = ccl.build([ccl.entry(**good())])
    assert "NOT CLAIMED" in led["parity_claim"]
    assert "NOT YET STUDIED" in led["coverage_honesty"]


def test_the_real_ledger_on_disk_is_honest():
    p = REPO / "receipts/headless/CUDA_CAPABILITY_LEDGER.json"
    led = json.loads(p.read_text())
    assert led["count"] >= 17
    assert led["classes_with_no_entry"] == []
    # This once asserted that unmeasured gaps OUTNUMBER measured ones, which was true
    # when written and stopped being true as the work progressed -- a transient fact
    # encoded as a law. The invariant that actually matters is that the two counts are
    # CONSISTENT WITH THE ENTRIES, so the ledger cannot overstate its own coverage.
    measured = sum(1 for e in led["entries"]
                   if isinstance(e["performance_gap"], dict)
                   and e["performance_gap"].get("measured"))
    assert led["performance_gaps_measured"] == measured
    assert led["performance_gaps_unmeasured"] == led["count"] - measured
    for e in led["entries"]:
        pg = e["performance_gap"]
        if isinstance(pg, dict) and pg.get("measured"):
            assert pg.get("receipt"), f"{e['capability_id']} claims a measured gap with no receipt"
    for e in led["entries"]:
        if e["semantic_gap"] == "NONE":
            assert e.get("evidence"), f"{e['capability_id']} claims parity with no evidence"


def test_margin_is_symmetric_between_wins_and_losses():
    """|speedup-1| capped a slowdown at 100% while a speedup was unbounded, so the
    instrument was quietly harder on losses than wins."""
    import bench
    fast = {"median_s": 1.0, "iqr_spread_pct": 14.0, "reliable": False}
    slow = {"median_s": 3.0, "iqr_spread_pct": 14.0, "reliable": False}
    loss = bench.compare({"b": fast, "c": slow}, baseline="b", candidate="c")
    win = bench.compare({"b": slow, "c": fast}, baseline="b", candidate="c")
    assert loss["margin_pct"] == win["margin_pct"] == 200.0
    assert loss["verdict"].startswith("BASELINE_WINS")
    assert win["verdict"].startswith("CANDIDATE_WINS")


def test_recount_repairs_a_drifted_ledger():
    """The drift that actually happened: an entry's performance_gap was changed to
    measured without recomputing, leaving the ledger claiming 10 when it had 11."""
    led = ccl.build([ccl.entry(**good())])
    led["performance_gaps_measured"] = 99
    led["count"] = 42
    ccl.recount(led)
    assert led["count"] == 1
    assert led["performance_gaps_measured"] == 0
    assert led["performance_gaps_unmeasured"] == 1


# --- G055 census expansion pins -------------------------------------------
# The census added 48 entries across PROGRAMMING MODEL, MEMORY, EXECUTION,
# COMPILER, MATH ECOSYSTEM and PROFILING. These pin that the same two
# disciplines the seed ledger enforced still hold at 68 entries, not just at
# the original 20 -- a validator that only gets exercised by the fixture in
# `good()` could pass while a real new-family entry sails through unchecked.


def test_new_family_entry_still_refuses_incomplete():
    """The validator must refuse an incomplete entry from one of the NEW
    families too, not just the `good()` fixture shape."""
    real = dict(ccl.NEW_ENTRIES[0])
    real.pop("cuda_mechanism")
    with pytest.raises(ccl.CCLError, match="missing"):
        ccl.entry(**real)


def test_new_family_measured_gap_without_receipt_still_refused():
    """One of the census's own measured entries, with its receipt stripped,
    must still be refused -- proving the discipline was applied by the
    validator and not just by the author's discipline."""
    measured_entries = [e for e in ccl.NEW_ENTRIES
                         if isinstance(e["performance_gap"], dict)
                         and e["performance_gap"].get("measured")]
    assert measured_entries, "expected at least one measured new entry to test against"
    broken = dict(measured_entries[0])
    broken["performance_gap"] = {"measured": True, "value": "some value"}
    with pytest.raises(ccl.CCLError, match="needs a receipt"):
        ccl.entry(**broken)


def test_census_expansion_covers_the_six_families_without_touching_multi_device():
    """MULTI_DEVICE.peer_access is blocked on hardware and this campaign's own
    rule is to keep it that way -- the census must add zero MULTI_DEVICE
    entries even though it adds 48 entries everywhere else."""
    led = ccl.census_ledger()
    assert led["count"] == 68
    assert led["by_class"]["MULTI_DEVICE"] == 1
    assert led["classes_with_no_entry"] == []
    # every CLASS gained at least one census-era entry except MULTI_DEVICE
    new_by_class: dict[str, int] = {}
    for e in ccl.NEW_ENTRIES:
        c = e["capability_id"].split(".")[0]
        new_by_class[c] = new_by_class.get(c, 0) + 1
    assert "MULTI_DEVICE" not in new_by_class
    assert set(new_by_class) == set(ccl.CLASSES) - {"MULTI_DEVICE"}


def test_census_ledger_never_writes_unqualified_cuda_parity():
    """The steer's own rule: never write 'CUDA parity' unqualified anywhere in
    the ledger. The one place the phrase legitimately appears is inside the
    ledger's own parity_claim statement NAMING the rule -- that mention is
    itself quoted and qualified by the sentence around it, so it is excluded
    here by construction rather than the check being a blind string search."""
    led = ccl.census_ledger()
    for e in led["entries"]:
        blob = json.dumps(e).lower()
        assert "cuda parity" not in blob, f"{e['capability_id']} writes the forbidden phrase"


def test_census_new_entries_gap_counts_match_the_receipt_plan():
    """Pins the specific split this campaign measured: 5 of the 48 new entries
    cite a real receipt as measured, the other 43 are honestly unmeasured. A
    silent change to either number means an entry's performance_gap was edited
    without the corresponding receipt work actually happening."""
    measured = sum(1 for e in ccl.NEW_ENTRIES
                    if isinstance(e["performance_gap"], dict)
                    and e["performance_gap"].get("measured"))
    assert measured == 5
    assert len(ccl.NEW_ENTRIES) - measured == 43
