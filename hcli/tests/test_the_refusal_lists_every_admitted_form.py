"""A refusal that omits an accepted form sends the scientist hunting.

`python <path>` has always been admitted, and the NOT_ADMITTED text never named
it. Measured on the daemon: the worker oscillated between two rejected shapes
for eight minutes and reached the working one by trial, not by being told.

The list and the admitter must not be able to drift apart, so this drives both:
every form the message advertises must actually be admitted, and the forms the
admitter accepts must be advertised.
"""
from __future__ import annotations

import re

from hcli.engine import Engine, _NOT_ADMITTED_REASON
from hcli.workspace import Workspace


class _Pool:
    model_path = "sealed-3.14"
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


def _engine(tmp_path):
    (tmp_path / "t.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    return Engine(Workspace(str(tmp_path)), runtime_provider=lambda: _Pool())


ACCEPTED = ["t.py", "pytest t.py", "python -m pytest t.py", "python t.py"]


def test_every_form_the_message_advertises_is_really_admitted(tmp_path):
    eng = _engine(tmp_path)
    for form in ACCEPTED:
        assert eng._admit_test(form).get("admitted") is True, form


def test_every_admitted_form_is_named_in_the_refusal(tmp_path):
    eng = _engine(tmp_path)
    for form in ACCEPTED:
        assert eng._admit_test(form).get("admitted") is True, form
        # The form with its path generalised: "python t.py" must appear as
        # python followed by SOME .py path, and must not be satisfied by the
        # "python -m pytest <path>" line that merely starts with the same word.
        pattern = r"\s+".join(
            r"\S+\.py" if tok.endswith(".py") else re.escape(tok)
            for tok in form.split()
        )
        assert re.search(pattern, _NOT_ADMITTED_REASON), (
            f"{form!r} is admitted but the refusal never shows that shape, so a "
            f"worker reading the refusal cannot find it: {_NOT_ADMITTED_REASON}"
        )


def test_a_refused_form_is_still_refused(tmp_path):
    eng = _engine(tmp_path)
    for form in ("tests.run(['t.py'])", "python -m unittest t.py", "pytest t.py | tee x"):
        assert eng._admit_test(form).get("admitted") is not True, form
