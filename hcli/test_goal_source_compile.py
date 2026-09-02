"""An oversized goal must be COMPILED, not truncated and not refused.

The sovereign ultragoal measured 12,456 tokens against 5,632 usable input on
sealed-3.14. `_build_model_payload` began with `del compiled` and then embedded
the raw text verbatim, so the Goal Compiler's output was computed and thrown
away and an oversized goal could only ever be refused by preflight.

Source is not active context: the exact bytes are persisted, hashed, and
referenced, and the resident retrieves spans with fs.read on demand.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hcli.engine import Engine, _CHARS_PER_TOKEN
from hcli.goal import GoalCompiler

PROFILE = "/Users/scammermike/Downloads/hawking/hcli/hawking-native.sealed-3.14.json"


class _Pool:
    model_path = PROFILE
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


class _Cfg:
    def model_tokens(self):
        return (None, None)


def _engine(root):
    eng = Engine.__new__(Engine)
    eng.root = Path(root)
    eng.runtime_provider = lambda: _Pool()
    eng.runtime_count = 1
    eng.config = _Cfg()
    return eng


SMALL = "Read hcli/paths.py and report the directory names it defines."
BIG = ("HAWKING SOVEREIGN ULTRAGOAL. Enter sovereign mutation mode. " * 400)


def test_a_goal_that_fits_is_sent_verbatim(tmp_path):
    eng = _engine(tmp_path)
    block = eng._goal_block(SMALL, GoalCompiler().compile(SMALL))
    assert block == f"GOAL:\n{SMALL}", "a small goal must not be rewritten"


def test_an_oversized_goal_is_compiled_not_truncated(tmp_path):
    eng = _engine(tmp_path)
    block = eng._goal_block(BIG, GoalCompiler().compile(BIG))
    raw_tokens = len(BIG) // _CHARS_PER_TOKEN
    block_tokens = len(block) // _CHARS_PER_TOKEN
    assert block_tokens < raw_tokens / 10, (
        f"{raw_tokens} raw tokens became {block_tokens}; not a compilation"
    )
    assert "compiled kernel" in block
    # Truncation is the failure mode this exists to prevent.
    assert not block.startswith(f"GOAL:\n{BIG[:200]}"), "goal was truncated, not compiled"


def test_the_exact_source_is_preserved_hashed_and_reachable(tmp_path):
    eng = _engine(tmp_path)
    block = eng._goal_block(BIG, GoalCompiler().compile(BIG))

    rel = [w for w in block.split() if w.startswith(".hcli/sources/")]
    assert rel, f"no source handle in the kernel:\n{block}"
    path = Path(tmp_path) / rel[0]
    assert path.is_file(), "source was referenced but never written"
    assert path.read_text() == BIG, "persisted source is not byte-exact"
    assert hashlib.sha256(BIG.encode()).hexdigest()[:16] in block, "hash not carried"


def test_the_kernel_fits_the_budget_the_raw_goal_blew(tmp_path):
    from hcli.context_budget import preflight

    eng = _engine(tmp_path)
    budget = eng._context_budget()
    block = eng._goal_block(BIG, GoalCompiler().compile(BIG))
    assert preflight(budget, len(BIG) // _CHARS_PER_TOKEN, kind="root").ok is False
    assert preflight(budget, len(block) // _CHARS_PER_TOKEN, kind="root").ok is True


def test_the_same_source_is_written_once(tmp_path):
    eng = _engine(tmp_path)
    compiled = GoalCompiler().compile(BIG)
    eng._goal_block(BIG, compiled)
    eng._goal_block(BIG, compiled)
    files = list((Path(tmp_path) / ".hcli" / "sources").glob("goal-*.txt"))
    assert len(files) == 1, f"content-addressed store wrote {len(files)} copies"


def test_no_compiled_ir_means_no_rewrite(tmp_path):
    """Callers that pass nothing keep the old behaviour exactly."""
    eng = _engine(tmp_path)
    assert eng._goal_block(BIG, None) == f"GOAL:\n{BIG}"


def test_build_model_payload_actually_uses_the_compiler(tmp_path):
    """The CALL SITE, not the helper.

    Mutating `_build_model_payload` to embed the raw goal again left every test
    above green, because they all called `_goal_block` directly. That is the
    same "verified in isolation, never on the live path" defect that produced
    `del compiled` in the first place. This drives the real Engine.
    """
    from hcli.workspace import Workspace

    eng = Engine(Workspace(str(tmp_path)), runtime_provider=lambda: _Pool())
    payload = eng._build_model_payload(BIG, [], GoalCompiler().compile(BIG))
    user = [m for m in payload["messages"] if m["role"] == "user"][0]["content"]

    assert "compiled kernel" in user, "the payload embedded the raw goal"
    assert BIG[:400] not in user, "raw source leaked into the posted payload"
    assert ".hcli/sources/" in user, "no retrievable source handle in the payload"
    assert len(user) // _CHARS_PER_TOKEN < len(BIG) // _CHARS_PER_TOKEN / 10


def test_a_small_goal_still_reaches_the_payload_verbatim(tmp_path):
    from hcli.workspace import Workspace

    eng = Engine(Workspace(str(tmp_path)), runtime_provider=lambda: _Pool())
    payload = eng._build_model_payload(SMALL, [], GoalCompiler().compile(SMALL))
    user = [m for m in payload["messages"] if m["role"] == "user"][0]["content"]
    assert SMALL in user and "compiled kernel" not in user
