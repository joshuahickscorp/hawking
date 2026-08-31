"""A scar without a regression test is a story. These check it is a guard.

Every autonomy defect this campaign found looked healthy from outside: a filter
that matched nothing, a counter of questions nobody answered, a gate certifying
itself, a status string standing in for a cause. None produced an error. That is
exactly why they are recorded rather than merely fixed.
"""
import json

from tools.future import autonomy_scars as asc
from tools.future._common import REPO, RECEIPTS


def test_every_scar_names_a_regression_test_that_exists_and_names_the_defect():
    missing = asc.missing_regression_tests()
    assert missing == [], f"scars with no guard: {missing}"


def test_every_scar_records_the_symptom_that_hid_it():
    """The fix is cheap to rediscover. The disguise is what costs a campaign."""
    for scar in asc.scars():
        assert scar["why_it_hid"], f"{scar['id']} does not say how it stayed hidden"
        assert scar["cost"], f"{scar['id']} does not say what it cost"
        assert scar["law"], f"{scar['id']} states no law"
        assert scar["claim_refuted"]


def test_the_metal_scar_stays_open_on_the_part_that_is_not_settled():
    """The host claim is falsified. The original process failure is not diagnosed.

    Closing the whole scar would quietly claim a diagnosis nobody has made.
    """
    scar = [s for s in asc.scars() if s["id"] == "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM"][0]
    assert "unidentified" in scar["reopen_condition"]
    assert "open work" in scar["reopen_condition"]


def test_the_lane_and_keying_scars_can_never_reopen():
    """Some laws have no legitimate exception; a reopen condition invites one."""
    for sid in ("STATIC_LANE_TAXONOMY_DIVERGED_FROM_FRONTIER",
                "SCAR_LOOKUP_ON_IMPLEMENTATION_NAMES",
                "DECLARED_CAPABILITY_READ_AS_EXECUTED_CAPABILITY"):
        scar = [s for s in asc.scars() if s["id"] == sid][0]
        assert scar["reopen_condition"].startswith("never")


def test_receipt_reports_missing_guards_rather_than_hiding_them():
    out = asc.build()
    doc = json.loads(out.read_text())
    assert doc["n_scars"] == len(asc.SCARS)
    assert "scars_without_a_regression_test" in doc
    assert doc["general_law"].startswith("STATUS LABELS ARE HYPOTHESES")
