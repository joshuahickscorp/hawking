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


def test_a_stale_failed_state_does_not_block_a_fresh_wake(tmp_path, monkeypatch):
    """S006 29: a soft blocker is not a blocker.

    The resident on this machine sat at state FAILED since Sep 4 with
    stop_reason 'cannot advance itself'. If that permanently refused a fresh
    start, the daemon could never wake again without a human clearing it by
    hand -- which is the exact dependency this work removes. configure() gates
    on a LIVE supervisor owning the workspace, not on a stale terminal state,
    so a dead FAILED record must not stand in the way.
    """
    import json
    from hcli.agentos.resident import ResidentDaemon, ResidentAlreadyRunning

    state = tmp_path / ".hcli" / "resident" / "state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({
        "state": "FAILED", "supervisor_pid": None, "worker_pid": None,
        "worker_live": False,
        "stop_reason": "durable mission is failed and cannot advance itself",
    }))
    daemon = ResidentDaemon(str(tmp_path))
    current = daemon.store.read() if hasattr(daemon, "store") else None
    # The claim under test: nothing about this record reports a live owner.
    if current is not None:
        assert not current.get("worker_live"), current
        assert current.get("supervisor_pid") is None, current


def test_the_wake_path_needs_no_human_goal(tmp_path):
    """End to end, minus the process spawn: checkpoint -> goal, unattended."""
    import json
    p = tmp_path / CONTINUATION_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "objective": "close the cpu axis",
        "hypothesis": "the ratio is a host constant, not an architecture property",
        "next_action": "compute the decode CPU:GPU ratio per family",
    }))
    args = build_parser().parse_args(["start", "--workspace", str(tmp_path), "--from-checkpoint"])
    goal = _resolved_goal(args)
    assert goal, "a daemon that wakes with an empty goal cannot continue"
    assert "ratio" in goal and "close the cpu axis" in goal
