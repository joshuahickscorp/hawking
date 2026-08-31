"""A dead unit is a fixture to refuse once, not inventory to re-serve.

live_catalog deliberately seeds closed scars so the refusal is exercised and the
scar id is NAMED - a required event, and hiding the ids would make them
unreachable. But the only path that RETIRED a dead unit was the model picking
it, and choose() no longer offers dead units at all. The fixtures became
permanent furniture.

Measured cost, from the 1816 s run: four units launched in the first 322 s, then
nothing for 1494 s. 61 refills served 5 distinct sets, one of them 54 times,
whose visible members are four WU.DEAD.* rows. A queue that cannot drain is not
a frontier, it is a wall, and every choose() after that asked one settled
question - which is also why divergence looked unmeasurable.
"""
from __future__ import annotations

from tools.future import model_bearing_torture as mbt


def test_the_catalog_still_seeds_dead_units():
    """The fixtures must stay. Removing them would make the scars unreachable."""
    dead = [r for r in mbt.live_catalog() if r.get("dead")]
    assert dead, "the negative control was deleted, not fixed"
    for row in dead:
        assert row.get("scar_id"), f"{row['id']} is dead with no scar id to name"
        assert row["scar_id"] in row["title"], "the scar id must reach the prompt"


def test_dead_units_are_recognised_as_dead():
    for row in mbt.live_catalog():
        if row.get("dead"):
            assert mbt.is_dead_unit(row), f"{row['id']} marked dead but not recognised"


def test_live_units_are_not_swept_up():
    live = [r for r in mbt.live_catalog() if not r.get("dead")]
    assert live, "the catalog is all fixtures and nothing can ever launch"
    for row in live:
        assert not mbt.is_dead_unit(row), f"{row['id']} is live but reads as dead"


def test_the_retirement_is_in_the_run_loop_not_only_the_pick_path():
    """Regression guard: the pick path cannot fire once choose() filters dead ids."""
    src = mbt.__file__
    text = open(src, encoding="utf-8").read()
    assert "dead_units_retired" in text
    # the retirement must be driven by the POLICY's refusals, which see every
    # dead unit, not by the model's pick, which now never contains one
    i = text.index("dead_units_retired")
    assert "policy_dead" in text[i - 1600:i], "retirement is not driven by policy refusals"
