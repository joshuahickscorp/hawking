"""The controls must stay honest: they must flip, and they must restore."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trial_negative_controls as nc
from _common import REPO


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_every_control_flips_the_judge():
    doc = nc.run()
    assert doc["n_controls"] >= 11
    assert doc["n_leaks"] == 0, doc["leaks"]


def test_controls_restore_every_receipt_they_mutate():
    """A control that corrupts a receipt and does not put it back is sabotage."""
    touched = [REPO / "receipts/future/MUTATION_ENGINE.json",
               REPO / "receipts/future/STATUS_CAUSALITY_CHALLENGE.json",
               REPO / "receipts/future/AUTONOMY_TIMELINE_30M.json"]
    before = {p: _digest(p) for p in touched}
    nc.run()
    for p in touched:
        assert _digest(p) == before[p], f"{p.name} was not restored"


def test_the_seal_control_is_the_edit_that_would_matter():
    doc = nc.run()
    seal = next(c for c in doc["controls"] if c["control"] == "seal_detects_the_elapsed_lie")
    assert seal["untouched_verifies"] is True
    assert seal["tampered_verifies"] is False


def test_the_bad_control_stays_recorded():
    """The first attempt emptied keys the evaluator never reads and cried leak.

    That belongs in the receipt, not in a commit message that ages out.
    """
    doc = nc.run()
    bad = doc["a_bad_control_of_my_own"]
    assert "historical_cases" in bad["why_it_was_wrong"]
    assert bad["the_lesson"]


def test_the_fixture_limitation_is_not_papered_over():
    lim = nc.run()["real_limitation_the_controls_do_not_fix"]
    assert lim["condition"] == "status_causality_challenged"
    assert lim["class"] == "fixture-only success"
    assert lim["what_would_close_it"]
