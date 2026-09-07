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
        reserve=2000,
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
        reserve=2000,
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


def test_the_reducer_accounts_for_what_the_contract_will_add():
    """The ladder fit a payload the post-contract preflight then refused.

    `contract.apply` injects the schema instruction -- about 713 tokens -- AFTER
    the ladder declares a fit. Measured live: the reducer approved a payload and
    the next check reported `demand 8739 exceeds per-request ctx 8192`.

    A reducer that shrinks against a size nobody posts is not a reducer.
    """
    import inspect

    from hcli.engine import Engine

    src = inspect.getsource(Engine._call_model)
    build_at = src.index("_schema_contract(")
    fit_at = src.index("_fit_payload_to_budget")
    assert build_at < fit_at, (
        "the contract must be built BEFORE fitting, so its cost can be reserved"
    )

    ladder = inspect.getsource(Engine._fit_payload_to_budget)
    assert "+ reserve" in ladder, "the reserve must enter the demand the ladder judges"


def test_a_reserve_makes_the_ladder_shed_sooner(tmp_path):
    """The load-bearing behaviour, not just the plumbing.

    Uses the real engine fixture and its real budget: a stub budget invites the
    test to disagree with production about what a budget even is.
    """
    eng = _engine(tmp_path)
    eng._estimate_prompt_tokens = lambda msgs: 5000

    def build(ev, cm, tr=""):
        return {"messages": [{"role": "user", "content": "x"}]}

    fits, _ = eng._fit_payload_to_budget(build, [], None, reserve=0)
    assert fits is not None, "5000 tokens alone must fit"

    with pytest.raises(ContextPreflightError):
        eng._fit_payload_to_budget(build, [], None, reserve=100_000)


def test_tool_history_is_compacted_when_the_next_round_overflows(tmp_path):
    """A real tool round must not bypass the reducer through ``history``."""
    eng = _engine(tmp_path)

    history = [
        {"role": "assistant", "content": "old assistant " + "x" * 9000},
        {"role": "user", "content": "old observation " + "x" * 9000},
        {"role": "assistant", "content": "latest assistant " + "x" * 9000},
        {"role": "user", "content": "latest observation " + "x" * 9000},
    ]

    def build(ev, cm, tr="", *, history=None):
        del ev, cm, tr
        return {
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "goal"},
                *(history or []),
            ]
        }

    eng._estimate_prompt_tokens = lambda messages: sum(
        len(str(item.get("content") or "")) // 4 for item in messages
    )
    payload, reduction = eng._fit_payload_to_budget(
        build, [], None, history=history
    )

    assert reduction is not None
    assert reduction["dropped_history"] == 2
    assert payload["messages"][-2]["content"].startswith("latest assistant")
    assert payload["messages"][-1]["content"].startswith("latest observation")


def test_tool_history_is_truncated_before_it_is_dropped(tmp_path):
    """A tight real packet keeps a bounded latest decision trace."""
    eng = _engine(tmp_path)
    history = [
        {"role": "assistant", "content": "assistant " + "a" * 9000},
        {"role": "user", "content": "observation " + "b" * 9000},
    ]

    def build(ev, cm, tr="", *, history=None):
        del ev, cm, tr
        return {
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "goal"},
                *(history or []),
            ]
        }

    # Model the already-large O003 base packet. The raw pair cannot fit, but
    # the bounded pair can; dropping it would make the next model turn repeat
    # the same tool request with no explanation of the prior result.
    eng._estimate_prompt_tokens = lambda messages: 4500 + sum(
        len(str(item.get("content") or "")) // 4
        for item in messages[2:]
    )
    payload, reduction = eng._fit_payload_to_budget(
        build, [], None, history=history
    )

    assert reduction is not None
    assert reduction["history_compacted"] is True
    assert [m["role"] for m in payload["messages"][-2:]] == ["assistant", "user"]
    assert "history content truncated from" in payload["messages"][-1]["content"]
def test_closed_tools_keep_evidence_and_newest_observation(tmp_path):
    """The final no-tools turn keeps causal evidence without growing forever."""
    eng = _engine(tmp_path)
    evidence = _evidence(tmp_path, 1, 200)
    trailing = eng._observations_block([
        {"tool": "fs.read", "ok": False, "text": "missing path"},
        {"tool": "fs.list", "ok": True, "text": "a very large listing" * 500},
    ], final=True)
    eng._tools_closed_for_round = True
    try:
        payload, reduction = eng._fit_payload_to_budget(
            lambda ev, cm, tr=trailing: eng._build_model_payload(
                "create the requested file",
                ev,
                None,
                context_memory=cm,
                trailing=tr,
            ),
            evidence,
            None,
            trailing=trailing,
        )
    finally:
        eng._tools_closed_for_round = False
    user = [m["content"] for m in payload["messages"] if m["role"] == "user"][0]
    assert "f0.txt" in user
    assert "OBSERVATIONS" in user
    assert "missing path" in user or "a very large listing" in user


def test_closed_turn_can_drop_evidence_and_old_observations_together(tmp_path):
    """The structured-output reserve must not make a fitting closed turn refuse."""
    eng = _engine(tmp_path)
    evidence_path = Path(tmp_path) / "frontier.md"
    evidence_path.write_text("measured frontier\n" + ("x" * 1500))
    evidence = [{"path": "frontier.md", "content": evidence_path.read_text()}]
    goal = "\n".join([
        "PHASE: running",
        "WORKUNIT: implement.repair",
        "ROLE: implementation",
        "OBJECTIVE: repair the measured implementation boundary",
        "FAILURE_CONTEXT: " + ("context refusal details " * 35),
        "ACCEPTANCE: an existing proving test must pass",
    ])
    trailing = eng._observations_block(eng._compact_closed_observations([
        {"tool": "fs.read", "ok": True, "text": "first result " + ("a" * 3000)},
        {"tool": "fs.read", "ok": True, "text": "second result " + ("b" * 3000)},
        {"tool": "fs.list", "ok": True, "text": "newest result " + ("c" * 3000)},
    ]), final=True)
    eng._tools_closed_for_round = True
    try:
        payload, reduction = eng._fit_payload_to_budget(
            lambda ev, cm, tr=trailing: eng._build_model_payload(
                goal, ev, None, context_memory=cm, trailing=tr
            ),
            evidence,
            None,
            trailing=trailing,
            reserve=2400,
        )
    finally:
        eng._tools_closed_for_round = False
    assert reduction is not None
    assert "evidence 0 + observations" in reduction["reduced_to"]
    user = [m["content"] for m in payload["messages"] if m["role"] == "user"][0]
    assert "OBSERVATIONS" in user
    assert "newest result" in user
