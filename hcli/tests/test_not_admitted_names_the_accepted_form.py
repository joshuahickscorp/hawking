"""A refusal that names no accepted form is a dead end.

Measured across four autonomy runs. The resident proposed its test command as
`tests.run(['hcli/tests/x.py'])` and as `python -m unittest ...`; both came back
as the bare string "NOT_ADMITTED". Nothing in that tells it which forms exist,
so it alternated between the same two rejected shapes and every unit died on
NO_EVIDENCE -- with a correct test file already written to disk.

The accepted set is narrow and knowable: a bare path, `pytest <path>`, or
`python -m pytest <path>`. Saying so costs one string and turns an unguessable
gate into a correctable one.
"""
from pathlib import Path

import pytest

from hcli.engine import Engine

REAL = "hcli/tests/test_not_admitted_names_the_accepted_form.py"


@pytest.fixture
def engine():
    e = Engine.__new__(Engine)
    e.root = Path(".").resolve()
    return e


@pytest.mark.parametrize("cmd", [
    "tests.run(['%s'])" % REAL,
    "python -m unittest %s" % REAL,
    "cat x | grep y",
    "",
])
def test_a_refused_command_names_what_would_be_accepted(engine, cmd):
    r = engine._admit_test(cmd)
    assert not r["admitted"]
    reason = str(r.get("reason") or "")
    assert "pytest" in reason and "bare path" in reason, (
        f"refusal gives the model nothing to correct toward: {reason!r}")


@pytest.mark.parametrize("cmd", [REAL, f"pytest {REAL}", f"python -m pytest {REAL} -q"])
def test_the_named_forms_are_actually_accepted(engine, cmd):
    """The guidance must not describe forms the gate then refuses."""
    assert engine._admit_test(cmd)["admitted"], (
        f"the refusal message advertises {cmd!r} but the gate rejects it")


def test_a_tool_call_is_still_refused(engine):
    """Naming the accepted forms must not widen what is admitted."""
    assert not engine._admit_test("tests.run(['x.py'])")["admitted"]
