"""The contract must publish exactly the forms _admit_test accepts.

Measured blocker: the resident wrote a correct file, proposed
`python -c "import probe; assert probe.VALUE == 7"`, and the harness refused the
FORM and deleted the work. The contract demanded tests without ever saying which
shapes were admissible -- the fifth time this session the harness scored the
model on something it had not communicated.
"""
import re
from hcli.engine import Engine  # noqa: F401  (import guards a syntax break)
from pathlib import Path

SRC = (Path(__file__).resolve().parent / "engine.py").read_text()


def test_contract_names_every_admissible_form():
    block = SRC[SRC.index("ADMISSIBLE TEST FORMS"):][:900]
    assert "pytest" in block
    assert "python -m pytest" in block
    assert re.search(r"path/to/test_x\.py", block), "bare-path form not published"
    assert re.search(r"python path/to/script\.py", block), "two-token form not published"


def test_contract_names_the_refused_form_that_actually_blocked_a_unit():
    block = SRC[SRC.index("ADMISSIBLE TEST FORMS"):][:900]
    assert 'python -c' in block, "the form that caused the measured rollback is not called out"


def test_contract_states_the_consequence():
    block = SRC[SRC.index("ADMISSIBLE TEST FORMS"):][:900]
    assert "ROLLED BACK" in block, "the model is not told the mutation is discarded"


def _engine():
    # _admit_test resolves the test path against self.root; nothing else on the
    # Engine is touched, so a bare instance with a root is the whole fixture.
    e = Engine.__new__(Engine)
    e.root = Path(__file__).resolve().parent.parent
    return e


def test_admit_test_still_refuses_python_dash_c():
    # The contract must not have been "fixed" by widening execution.
    got = Engine._admit_test(_engine(), 'python -c "import os; os.system(1)"')
    assert got["admitted"] is False, "arbitrary inline code became admissible"


def test_admit_test_still_refuses_shell_composition():
    for bad in ("pytest a.py && rm -rf /", "cat x | sh", "cd /tmp; pytest a.py"):
        assert Engine._admit_test(_engine(), bad)["admitted"] is False, bad


def test_admit_test_still_accepts_the_published_forms():
    for form in ("hcli/test_engine_tool_loop.py",
                 "pytest hcli/test_engine_tool_loop.py",
                 "python -m pytest hcli/test_engine_tool_loop.py"):
        assert Engine._admit_test(_engine(), form)["admitted"] is True, form
