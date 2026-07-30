#!/usr/bin/env python3.12
"""Fail when a load-bearing gate runner is deleted.

Three separate density passes have removed the tool that *judges* something
while leaving the receipt that says it passed:

  * the 11 GLM gate controllers      (recoverable at 791ced2c^)
  * the Q0 clean-container harness   (recovered from f2bed147)
  * the execution-grounded thesis gate runner

Each deletion was individually reasonable, since nothing referenced the file.
That is precisely the failure: a gate runner has no callers by construction, so
"no references" is not evidence it is dead. This test is the reference.

Adding a runner here is a claim that a receipt somewhere depends on it. Removing
one means the receipts it produced are no longer reproducible, which is a
decision to take deliberately rather than to discover later.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# path -> why its absence would invalidate a receipt
LOAD_BEARING = {
    "tools/eval/thesis_gate.py":
        "produces reports/eval/thesis_gate_*.json; without it the 7B, 14B and 32B "
        "quality numbers stop being reproducible or comparable",
    "tools/verify/blackbox.py":
        "runs the behaviour-constitution matrix; the BC-* pass counts come from here",
    "tools/adapters/verify_grades.py":
        "the only check that an adapter family does not claim a grade its evidence "
        "cannot support",
    "ramanujan/container/replay_capsule.sh":
        "replays a proof capsule in a pinned clean container; it is the whole content "
        "of the Q0 reproducibility claim",
    "ramanujan/container/pins.json":
        "the offline lock itself: Lean, Mathlib, elan, z3, cadical and the base image "
        "digest. Without it 'pinned' means nothing",
    "tools/graph/hawking_graph.py":
        "regenerates the semantic graph the topology analyses and LOC authority read",
}


def test_load_bearing_gate_runners_exist() -> None:
    missing = [
        f"{rel}\n      why it matters: {why}"
        for rel, why in sorted(LOAD_BEARING.items())
        if not (ROOT / rel).exists()
    ]
    assert not missing, (
        "a gate runner was deleted; the receipts it produced are no longer "
        "reproducible:\n    " + "\n    ".join(missing)
    )


def test_every_entry_is_actually_load_bearing() -> None:
    """Keep the list honest in the other direction: a path listed here that is
    empty is worse than no entry, because it makes the guard look effective."""
    empty = [rel for rel in LOAD_BEARING if (ROOT / rel).exists() and (ROOT / rel).stat().st_size == 0]
    assert not empty, f"listed but empty, so the guard proves nothing: {empty}"


if __name__ == "__main__":
    test_load_bearing_gate_runners_exist()
    test_every_entry_is_actually_load_bearing()
    print(f"ok: {len(LOAD_BEARING)} gate runners present")
