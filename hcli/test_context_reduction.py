"""An oversized turn must SHRINK before it is refused.

`preflight` raised ContextPreflightError straight into `resources.py`, which
grades it IMPOSSIBLE_CONTRACT. So one turn that did not fit ended the goal with
no attempt to recover — the single most likely way an unattended overnight run
dies at 3am. Deterministic evidence is re-readable with fs.read and the durable
checkpoint is re-readable from mission state, so both are droppable; the goal
is not, and is already compiled with its exact source on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hcli.engine import ContextPreflightError, Engine
from hcli.workspace import Workspace

PROFILE = "/Users/scammermike/Downloads/hawking/hcli/hawking-native.sealed-3.14.json"


class _Pool:
    model_path = PROFILE
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


def _engine(tmp_path):
    return Engine(Workspace(str(tmp_path)), runtime_provider=lambda: _Pool())


def _evidence(tmp_path, n, chars):
    """Real files: `_assert_evidence_fresh` re-reads every path from disk."""
    out = []
    for i in range(n):
        body = "x" * chars
        (Path(tmp_path) / f"f{i}.txt").write_text(body)
        out.append({"path": f"f{i}.txt", "content": body})
    return out


def test_a_turn_that_fits_is_not_reduced(tmp_path):
    eng = _engine(tmp_path)
    payload, reduction = eng._fit_payload_to_budget(
        lambda ev, cm: eng._build_model_payload("small goal", ev, None, context_memory=cm),
        _evidence(tmp_path, 2, 200),
        None,
    )
    assert reduction is None, "a fitting turn must be sent untouched"


def test_oversized_evidence_is_dropped_not_the_goal(tmp_path):
    eng = _engine(tmp_path)
    goal = "Report the number of layers."
    payload, reduction = eng._fit_payload_to_budget(
        lambda ev, cm: eng._build_model_payload(goal, ev, None, context_memory=cm),
        _evidence(tmp_path, 40, 4000),          # ~53k tokens of evidence, budget is 5632
        None,
    )
    assert reduction is not None, "an oversized turn must report its reduction"
    assert reduction["dropped_evidence"] > 0
    user = [m for m in payload["messages"] if m["role"] == "user"][0]["content"]
    assert goal in user, "the goal must survive reduction"


def test_the_reduced_payload_actually_fits(tmp_path):
    from hcli.context_budget import preflight

    eng = _engine(tmp_path)
    payload, reduction = eng._fit_payload_to_budget(
        lambda ev, cm: eng._build_model_payload("g", ev, None, context_memory=cm),
        _evidence(tmp_path, 40, 4000),
        None,
    )
    demand = eng._estimate_prompt_tokens(payload["messages"])
    assert preflight(eng._context_budget(), demand, kind="root").ok, (
        f"reduction returned a payload that still does not fit ({demand} tokens)"
    )


def test_an_irreducible_turn_still_refuses_honestly(tmp_path):
    """Reduction is not a licence to send something that cannot fit."""
    eng = _engine(tmp_path)
    huge_goal = "y" * 400_000          # ~133k tokens, no evidence to drop
    with pytest.raises(ContextPreflightError) as caught:
        eng._fit_payload_to_budget(
            lambda ev, cm: eng._build_model_payload(huge_goal, ev, None, context_memory=cm),
            [],
            None,
        )
    assert caught.value.shortfall > 0
    assert caught.value.remedy


def test_reduction_prefers_evidence_over_the_checkpoint(tmp_path):
    """Evidence goes first: it is the cheapest thing to re-read."""
    eng = _engine(tmp_path)
    payload, reduction = eng._fit_payload_to_budget(
        lambda ev, cm: eng._build_model_payload("g", ev, None, context_memory=cm),
        _evidence(tmp_path, 8, 3000),
        None,
    )
    assert reduction is not None
    assert "evidence" in reduction["reduced_to"]


def test_reduction_is_gradual_not_all_or_nothing(tmp_path):
    """Keep what fits. Collapsing straight to zero evidence is a regression.

    With a one-rung ladder the final fallback still "succeeds" by dropping
    everything, so an outcome-only assertion cannot tell a graded ladder from a
    cliff. This pins the granularity.
    """
    eng = _engine(tmp_path)
    # Sized so the full set overflows but a fraction of it fits comfortably.
    ev = _evidence(tmp_path, 8, 3000)
    payload, reduction = eng._fit_payload_to_budget(
        lambda e, cm: eng._build_model_payload("g", e, None, context_memory=cm),
        ev,
        None,
    )
    assert reduction is not None, "this fixture must overflow to be meaningful"
    assert reduction["dropped_evidence"] < len(ev), (
        "all evidence was dropped when a subset would have fit; "
        "the reduction ladder collapsed to a cliff"
    )
    user = [m for m in payload["messages"] if m["role"] == "user"][0]["content"]
    assert "=====" in user, "no evidence survived a reduction that should be partial"


def test_the_reducer_is_actually_on_the_model_call_path(tmp_path):
    """The CALL SITE. Bypassing the reducer in `_call_model` left every test
    above green, because they all drive `_fit_payload_to_budget` directly."""
    import inspect

    src = inspect.getsource(Engine._call_model)
    assert "_fit_payload_to_budget" in src, (
        "_call_model no longer reduces before preflight; an oversized turn "
        "will be refused outright again"
    )
