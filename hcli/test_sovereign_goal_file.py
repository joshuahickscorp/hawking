"""`civilization/sovereign-goal.txt` is what launches the daemon. It must carry every gate.

The file on disk was a 2810-char variant with no G001-G015 in it at all, while
the live daemon config held a 5444-char goal that had them. A launch from the
file would have silently dropped the entire obligation ledger and then reported
progress against a goal nobody chose.

`--goal-file` did not exist either, so the documented launch command could not
run and the goal's only durable home was a JSON field inside the daemon's own
state. This file guards the repaired arrangement: one reviewable goal file, and
a flag that reads it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from hcli.agentos.resident import build_parser

REPO = Path(__file__).resolve().parents[1]
GOAL_FILE = REPO / "civilization" / "sovereign-goal.txt"


@pytest.fixture(scope="module")
def goal_text():
    assert GOAL_FILE.is_file(), f"{GOAL_FILE} is missing"
    return GOAL_FILE.read_text(encoding="utf-8")


@pytest.mark.parametrize("gate", [f"G{n:03d}" for n in range(1, 16)])
def test_every_gate_is_named(goal_text, gate):
    assert gate in goal_text, (
        f"{gate} is absent from civilization/sovereign-goal.txt; launching from this file "
        "would drop that obligation without saying so"
    )


def test_every_named_verifier_exists(goal_text):
    """A gate pointing at a file that is not there is an unenforceable obligation."""
    named = sorted(set(re.findall(r"Verified by (\S+\.py)", goal_text)))
    assert len(named) >= 13, f"only {len(named)} verifiers named: {named}"
    missing = [p for p in named if not (REPO / p).is_file()]
    assert not missing, f"goal names verifiers that do not exist: {missing}"


def test_goal_file_and_goal_are_mutually_exclusive_and_one_is_required():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["replace"])
    with pytest.raises(SystemExit):
        parser.parse_args(["replace", "--goal", "x", "--goal-file", str(GOAL_FILE)])


def test_goal_file_is_actually_read(tmp_path):
    from hcli.agentos.resident import _resolved_goal

    path = tmp_path / "g.txt"
    path.write_text("a goal from a file\n", encoding="utf-8")
    args = build_parser().parse_args(["start", "--goal-file", str(path)])
    assert _resolved_goal(args) == "a goal from a file\n"
