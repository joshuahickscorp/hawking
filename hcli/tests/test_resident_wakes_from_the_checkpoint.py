"""S006 34: a daemon that needs a human to retype its goal cannot continue itself.

The resident's goal was reachable only two ways -- `--goal` or `--goal-file`,
mutually exclusive and required -- so "wake up and carry on" was impossible by
construction, no matter what state was on disk. campaign.checkpoint (G025)
already records an explicit NEXT ACTION; this wires the two together.

It refuses loudly on a missing or empty checkpoint. A daemon that starts on a
guessed objective is worse than one that will not start.
"""
import json
from pathlib import Path

import pytest

from hcli.agentos.resident import (
    CONTINUATION_REL,
    _goal_from_checkpoint,
    _resolved_goal,
    build_parser,
)


def _write(ws: Path, doc: dict) -> None:
    p = ws / CONTINUATION_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))


def test_the_flag_exists_and_is_exclusive_with_the_other_two(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["start", "--workspace", str(tmp_path), "--from-checkpoint"])
    assert args.from_checkpoint is True
    with pytest.raises(SystemExit):
        parser.parse_args(["start", "--from-checkpoint", "--goal", "x"])


def test_it_carries_the_next_action_through(tmp_path):
    _write(tmp_path, {
        "objective": "close the gpu axis",
        "active_specimen": "Qwen3-0.6B",
        "hypothesis": "h",
        "next_action": "measure the cpu axis on bodies under 8 GiB",
    })
    goal = _goal_from_checkpoint(tmp_path)
    assert goal.startswith("measure the cpu axis")
    assert "close the gpu axis" in goal
    assert "Qwen3-0.6B" in goal


def test_resolved_goal_routes_to_the_checkpoint(tmp_path):
    _write(tmp_path, {"objective": "o", "hypothesis": "h", "next_action": "do the thing"})
    args = build_parser().parse_args(
        ["start", "--workspace", str(tmp_path), "--from-checkpoint"])
    assert _resolved_goal(args).startswith("do the thing")


def test_it_refuses_rather_than_inventing_a_goal(tmp_path):
    with pytest.raises(SystemExit) as e:
        _goal_from_checkpoint(tmp_path)
    assert "no checkpoint" in str(e.value)


@pytest.mark.parametrize("bad", [{}, {"next_action": ""}, {"next_action": "   "}])
def test_a_checkpoint_that_cannot_say_what_to_do_is_refused(tmp_path, bad):
    _write(tmp_path, {"objective": "o", "hypothesis": "h", **bad})
    with pytest.raises(SystemExit) as e:
        _goal_from_checkpoint(tmp_path)
    assert "next_action" in str(e.value)


def test_the_two_original_paths_still_work(tmp_path):
    p = build_parser()
    assert _resolved_goal(p.parse_args(["start", "--goal", "inline"])) == "inline"
    f = tmp_path / "g.txt"
    f.write_text("from a file")
    assert _resolved_goal(p.parse_args(["start", "--goal-file", str(f)])) == "from a file"
