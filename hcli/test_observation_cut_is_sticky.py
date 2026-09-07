"""The observation cut must not slide, or the resident's KV prefix is worthless.

Observations are shed oldest-first, which is right: the newest tool result is
the one the model asked for. But shedding "the last N" re-cut the block at a
DIFFERENT observation every turn, because every turn appends one. The rendered
prompt therefore changed at the exact point observations begin, and the
resident could reuse its prefix only up to there.

Measured on one goal before this: five consecutive calls pinned at 1398 reused
tokens -- the system prompt, the tool schemas and the goal, nothing more --
while the prompts grew past 4700. Every token past 1398 was re-stepped at 580
GPU dispatches each, on 98% GPU-bound calls of 60 to 190 seconds.

Keeping an ABSOLUTE floor that only advances makes each turn the previous turn
plus an append, which is the only shape a prefix cache can reuse.
"""
from __future__ import annotations

from pathlib import Path

from hcli.engine import Engine
from hcli.workspace import Workspace

PROFILE = "sealed-3.14"


class _Pool:
    model_path = PROFILE
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


def _engine(tmp_path):
    return Engine(Workspace(str(tmp_path)), runtime_provider=lambda: _Pool())


def _text(payload):
    return "\n".join(str(m.get("content") or "") for m in (payload.get("messages") or []))


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_a_new_goal_starts_with_a_fresh_observation_cut(tmp_path):
    eng = _engine(tmp_path)
    eng._observation_floor = 7
    eng._reset_evidence_efficiency()
    assert eng._observation_floor == 0


def _run(tmp_path, monkeypatch, turns=24, chars=4000):
    """Drive the reducer the way the agent loop does: append, then rebuild.

    Under the test default budget (24,576 usable) nothing sheds and the fixture
    measures nothing. Production runs at 5,632. Pin a budget in that regime or
    this file is decoration.
    """
    monkeypatch.setenv("HCLI_CTX_SIZE", "12288")  # -> 4,096 usable
    eng = _engine(tmp_path)
    observations, prompts, cuts = [], [], []
    for i in range(turns):
        observations.append(
            {"tool": f"tool{i}", "ok": True, "text": f"result-{i} " + "y" * chars}
        )
        trailing = eng._observations_block(observations)
        payload, reduction = eng._fit_payload_to_budget(
            # The live caller CLOSES OVER trailing and the reducer overrides it
            # only on the shedding rungs. A default of "" here would silently
            # drop the block under test and the fixture would measure nothing.
            lambda ev, cm, tr=trailing: eng._build_model_payload(
                "make the change", ev, None, context_memory=cm, trailing=tr
            ),
            [],
            None,
            trailing=trailing,
        )
        prompts.append(_text(payload))
        cuts.append((reduction or {}).get("observation_floor", 0))
    return prompts, cuts


def test_the_cut_only_ever_moves_forward(tmp_path, monkeypatch):
    _, cuts = _run(tmp_path, monkeypatch)
    assert cuts == sorted(cuts), f"the cut rewound, re-admitting a shed result: {cuts}"


def test_the_cut_moves_less_often_than_a_sliding_window(tmp_path, monkeypatch):
    """The whole point, held to a MEASURED bar rather than a hopeful one.

    Measured on this fixture: the sliding window it replaces re-cuts on 21 of
    23 turns; the monotone floor re-cuts on 15. Under this much pressure the
    window genuinely must advance -- only two observations fit at a time -- so
    the win is the six turns that keep their prefix, not all of them.
    """
    prompts, cuts = _run(tmp_path, monkeypatch)
    assert cuts[-1] > 0, "this fixture must actually drive the reducer into shedding"
    moves = sum(1 for a, b in zip(cuts, cuts[1:]) if a != b)
    assert moves <= 18, (
        f"the cut moved on {moves} of {len(cuts) - 1} turns, which is sliding-"
        f"window behaviour; each move invalidates the resident's prefix: {cuts}"
    )


def test_the_prompt_grows_by_append_once_shedding_has_settled(tmp_path, monkeypatch):
    """A turn that only appends is a turn the resident can prefill from cache."""
    prompts, cuts = _run(tmp_path, monkeypatch)
    settled = [
        (a, b) for a, b, ca, cb in zip(prompts, prompts[1:], cuts, cuts[1:]) if ca == cb
    ]
    assert settled, "no two consecutive turns shared a cut"
    for before, after in settled:
        assert after.startswith(before[: _shared_prefix(before, after)])
        assert _shared_prefix(before, after) >= len(before) * 0.9, (
            "consecutive turns at the same cut diverged early; the prompt is "
            "being rewritten, not appended to"
        )
