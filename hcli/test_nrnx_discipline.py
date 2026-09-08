"""S012 §102: never call a mutable candidate NX, never call the executable NR."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
from nrnx_discipline import violations, nx_claim_is_admissible, NX_PROMOTION_CONDITIONS, scan


def test_asserting_nx_status_is_flagged():
    for bad in ("This is the NX.", "We produced an NX today.",
                "the candidate was promoted to NX", "NX is complete"):
        assert violations(bad), f"not flagged: {bad!r}"


def test_discussing_the_distinction_is_not_a_violation():
    ok = ("NX is the frozen executable result and NR is the mutable representation; "
          "this candidate is an NR and is not NX because it has no promotion evidence.")
    assert violations(ok) == [], violations(ok)


def test_a_claim_without_every_condition_is_inadmissible():
    rec = {c: True for c in NX_PROMOTION_CONDITIONS}
    ok, missing = nx_claim_is_admissible(rec)
    assert ok and not missing
    for drop in ("no_hidden_dense_parent", "oiii_adversarial_qualification",
                 "complete_persistent_accounting"):
        partial = dict(rec); partial[drop] = False
        ok, missing = nx_claim_is_admissible(partial)
        assert not ok and drop in missing, drop


def test_all_ten_conditions_are_required():
    assert len(NX_PROMOTION_CONDITIONS) == 10, NX_PROMOTION_CONDITIONS
    assert nx_claim_is_admissible({})[0] is False


def test_the_campaign_corpus_asserts_no_nx():
    """Nothing in this campaign may claim NX. Nothing has been promoted."""
    root = Path(__file__).resolve().parent.parent
    paths = list((root / "receipts" / "future").glob("O003*.json"))
    paths += list((root / "receipts" / "future").glob("MOE_*.json"))
    hits = scan(paths)
    assert hits == [], f"corpus asserts NX status: {hits[:3]}"
