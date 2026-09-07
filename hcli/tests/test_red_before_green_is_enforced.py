"""A proving test that was already green must not accept the mutation.

Measured 2026-09-06. An HCLI mission was asked to add two measured stages to
tools/sovereign/g002_overhead.py and to write a proving test. It produced:

    tools/sovereign/test_g002_attribution.py:
        import g002_overhead
        def test_g002():
            assert g002_overhead

and NEVER CHANGED THE PRODUCER. That test passes identically with and without
the mutation. The verifier accepted it.

_compute_red_before_green already COMPUTES whether the named proving test was
red before the mutation. Nothing enforced it: engine.py hard-coded

    result["red_before_green_advisory"] = True

so acceptance keyed only on the post-mutation tests being green. A proving
test that PASSES against the pre-mutation tree is not evidence of the
mutation. It is now refused with reason NOT_RED_BEFORE, which is not
NO_EVIDENCE (empty/missing tests still use that reason and still keep the
bytes as unverified).
"""
from __future__ import annotations

import json
from pathlib import Path

from hcli.engine import Engine
from hcli.events import EventBus
from hcli.workspace import Workspace


WRONG_ADD = "def add(a, b):\n    return a * b - 999\n"
RIGHT_ADD = "def add(a, b):\n    return a + b\n"
TEST_ADD = (
    "from calc import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)

PRODUCER = "VALUE = 1\n"
PRODUCER_PLUS = "VALUE = 1\n\ndef extra():\n    return 2\n"
IMPORT_SMOKE = (
    "import producer\n\n"
    "def test_g002():\n"
    "    assert producer\n"
)

G002 = "def stage_one():\n    return 1\n"
G002_SMOKE = (
    "import g002_overhead\n\n"
    "def test_g002():\n"
    "    assert g002_overhead\n"
)


def _engine(tmp_path):
    return Engine(
        workspace=Workspace(str(tmp_path)),
        event_bus=EventBus(),
        runtime_count=1,
        model_name="/m.gguf",
    )


def _receipt_validation(result):
    receipt = json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))
    return receipt["validation"]


def test_already_green_proving_test_is_refused_not_red_before(tmp_path):
    """The load-bearing gate: a test that already passed cannot prove a mutation."""
    producer = tmp_path / "producer.py"
    test_file = tmp_path / "test_producer.py"
    producer.write_text(PRODUCER, encoding="utf-8")
    test_file.write_text(IMPORT_SMOKE, encoding="utf-8")
    engine = _engine(tmp_path)

    pre = engine._validate([producer, test_file], ["test_producer.py"])
    assert pre.get("ok") is True, pre
    assert engine._test_record_passed(
        [c for c in pre["checks"] if c.get("kind") == "test"][0]
    )

    producer.write_text(PRODUCER_PLUS, encoding="utf-8")
    post = engine._validate(
        [producer, test_file],
        ["test_producer.py"],
        pre_mutation=pre,
    )
    assert post.get("ok") is False, (
        "a proving test that was already green was accepted as evidence; "
        "red_before_green was advisory and did not bind"
    )
    assert post.get("reason") == "NOT_RED_BEFORE", post
    assert post.get("red_before_green") is False
    assert post.get("red_before_green_advisory") is True, (
        "the advisory field stays on the receipt for continuity; it is not "
        "the acceptance decision"
    )
    assert post.get("reason") != "NO_EVIDENCE"


def test_a_genuinely_red_before_green_mutation_is_still_accepted(tmp_path):
    """Negative control: a test that was red, then green, must still complete."""
    (tmp_path / "calc.py").write_text(WRONG_ADD, encoding="utf-8")
    (tmp_path / "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
    engine = _engine(tmp_path)
    engine._call_model = lambda p, e, c, **kwargs: {
        "kind": "mutation",
        "content": "fix add",
        "operations": [
            {
                "op": "replace",
                "path": "calc.py",
                "old_text": "return a * b - 999",
                "new_text": "return a + b",
            }
        ],
        "tests": ["test_calc.py"],
    }
    result = engine.execute("fix add")
    validation = _receipt_validation(result)
    assert result["status"] == "completed", result
    assert validation.get("ok") is True, validation
    assert validation.get("red_before_green") is True, validation
    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == RIGHT_ADD


def test_import_smoke_against_unchanged_producer_is_refused(tmp_path):
    """The exact measured scenario: new import-smoke, producer never changed."""
    sovereign = tmp_path / "tools" / "sovereign"
    sovereign.mkdir(parents=True)
    (sovereign / "g002_overhead.py").write_text(G002, encoding="utf-8")
    engine = _engine(tmp_path)
    engine._call_model = lambda p, e, c, **kwargs: {
        "kind": "mutation",
        "content": "prove g002",
        "operations": [
            {
                "op": "create",
                "path": "tools/sovereign/test_g002_attribution.py",
                "new_text": G002_SMOKE,
            }
        ],
        "tests": ["tools/sovereign/test_g002_attribution.py"],
    }
    result = engine.execute("add measured stages and a proving test")
    validation = _receipt_validation(result)
    assert result["status"] != "completed", result
    assert validation.get("ok") is not True, validation
    assert validation.get("reason") == "NOT_RED_BEFORE", validation
    assert validation.get("reason") != "NO_EVIDENCE"
    assert not (sovereign / "test_g002_attribution.py").exists(), (
        "the vacuous proving test was kept; a refused mutation must roll back"
    )
    assert (sovereign / "g002_overhead.py").read_text(encoding="utf-8") == G002


def test_unknown_before_state_fails_closed(tmp_path):
    """If the before-state cannot be established, accept is forbidden."""
    producer = tmp_path / "producer.py"
    test_file = tmp_path / "test_producer.py"
    producer.write_text(PRODUCER_PLUS, encoding="utf-8")
    test_file.write_text(IMPORT_SMOKE, encoding="utf-8")
    engine = _engine(tmp_path)
    post = engine._validate(
        [producer, test_file],
        ["test_producer.py"],
        pre_mutation={"ok": False, "checks": []},
    )
    assert post.get("ok") is False, post
    assert post.get("reason") == "RED_BEFORE_UNESTABLISHED", post
    assert post.get("red_before_green") is None
    assert post.get("reason") != "NO_EVIDENCE"
