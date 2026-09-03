"""Can the roadmap ANSWER the operator's eleven questions?

The operator's acceptance test is to hand a fresh model only
H-ROADMAP-REVISED.md plus the repo and ask eleven questions. Claude CANNOT grade
that -- it knows what the answers should be, which disqualifies it as examiner,
and a model grading its own document is the self-consistency trap this campaign
exists to refuse.

What CAN be checked mechanically is weaker but real: does the document CONTAIN
an answer to each question, in a form a stranger could find? A missing section is
a guaranteed failure of the real test; a present one is necessary, not sufficient.

This is a floor, not the acceptance test. It never substitutes for it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROADMAP = Path.home() / "Downloads" / "H-ROADMAP-REVISED.md"


def _text() -> str:
    if not ROADMAP.is_file():
        pytest.skip("the revised roadmap has not been emitted on this host")
    return ROADMAP.read_text()


# (question, markers that must ALL appear). Markers are chosen so a stranger
# scanning for the answer would land on the right section.
QUESTIONS = [
    ("WHAT IS HAWKING?",
     ["SELF-OPTIMIZING PHYSICAL AI COMPUTER", "five eras", "sovereign control plane"]),
    ("WHAT EXISTS?",
     ["PART I", "Status census", "Evidence coverage"]),
    ("WHAT IS ONLY SIMULATED?",
     ["FUNCTIONAL_SIM", "Simulated is not measured"]),
    ("WHAT IS PHYSICALLY PROVEN?",
     ["PHYSICALLY_MEASURED", "HARDWARE_MEASURED"]),
    ("WHAT IS RUNNING?",
     ["current mission / resident state", "active background jobs"]),
    ("WHAT IS STALE?",
     ["STALE_IF", "valid_for_head", "roadmap_state_sha256"]),
    ("WHAT IS THE HIGHEST-LEVERAGE NEXT ACTION?",
     ["HOT OPERATIONAL FRONTIER", "unlocks transitive"]),
    ("WHO OWNS IT?",
     ["owner", "CHECK OWNERSHIP"]),
    ("WHAT WILL IT UNLOCK?",
     ["unlocks direct", "unlocks transitive"]),
    ("HOW WILL IT BE VERIFIED?",
     ["verifier", "stop condition"]),
    ("WHAT SHOULD IT NOT TOUCH?",
     ["hcli/", "Never reorganize underneath an active Odyssey"]),
]


@pytest.mark.parametrize("question,markers", QUESTIONS, ids=[q for q, _ in QUESTIONS])
def test_the_document_contains_an_answer_to(question, markers):
    text = _text()
    missing = [m for m in markers if m not in text]
    assert not missing, (
        f"a fresh model asking {question!r} would find nothing for: {missing}"
    )


def test_the_answers_appear_before_the_bulk_so_a_stranger_reaches_them():
    """Orientation must precede detail, or the reader drowns before orienting."""
    text = _text()
    head = text[: text.index("# PART I")] if "# PART I" in text else text
    for marker in ("SELF-OPTIMIZING PHYSICAL AI COMPUTER", "READ IN THIS ORDER",
                   "HOT OPERATIONAL FRONTIER", "STALE_IF"):
        assert marker in head, f"{marker!r} appears only after PART I"


def test_this_is_not_the_operators_acceptance_test():
    """Recorded in the suite so nobody mistakes the floor for the bar.

    The real test is a FRESH model, given only this file and the repo, answering
    the eleven questions and choosing the same frontier as the live campaign.
    This checks only that an answer is present to be found.
    """
    assert len(QUESTIONS) == 11
