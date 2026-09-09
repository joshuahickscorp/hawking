"""A test run's OUTCOME must reach the trace, not just the fact that it ran.

Same defect shape as the mutation verdict, one tool over. `ok` on a `tests.run`
entry means the runner executed; it says nothing about whether the tests passed.
So a supervisor reading the trace cannot tell a RED from a GREEN, and campaign
cycle 79 proved that is not hypothetical: the body labelled

    "5 passed in 0.06s (returncode 0)"

as `RED:`, and the phase machine advanced to ACT_REQUIRED on it — demanding an
edit to fix a defect its own discriminator had just shown does not exist. The
body self-corrected at the next cycle and refused to mutate. The false-positive
class did not correct itself.

`_tests_run` already returns `verified` and `returncode`. The trace threw both
away. This pins that it does not.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.chat_tools import run_with_tools

REPO = str(Path(__file__).resolve().parents[2])


def _script(*texts):
    out = list(texts)

    def complete(_conversation):
        return out.pop(0) if out else "done"
    return complete


class _Pool:
    model_path = "sealed-3.14"
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


def _run(tmp, paths, body="def test_ok():\n    assert True\n"):
    """An ENGINE is required, not decoration: tests.run is a BUILDER tool, so
    without write authority the menu refuses it with "not offered to chat" and
    the trace entry is that refusal rather than a run. The first version of this
    test asserted against the refusal and looked like a missing field."""
    (Path(tmp) / "test_probe.py").write_text(body)
    from hcli.engine import Engine
    from hcli.tool_registry import default_tool_registry
    from hcli.workspace import Workspace
    reg = default_tool_registry(tmp, repo_root=tmp)
    engine = Engine(Workspace(tmp), runtime_provider=lambda: _Pool())
    call = json.dumps({"tool": "tests.run",
                       "arguments": {"runner": "pytest", "root": tmp, "paths": paths}})
    _, trace = run_with_tools(_script(call), [], reg, engine=engine)
    runs = [e for e in trace if e["tool"] == "tests.run"]
    assert runs and runs[0].get("error") != "not offered to chat", (
        f"tests.run was not offered: {runs}")
    return runs


class TestTheTraceSaysWhetherTestsPassed(unittest.TestCase):
    def test_a_passing_run_is_marked_verified(self):
        with TemporaryDirectory() as tmp:
            runs = _run(tmp, ["test_probe.py"])
            self.assertEqual(len(runs), 1, "tests.run did not reach the trace")
            e = runs[0]
            self.assertEqual(e.get("returncode"), 0,
                             f"the returncode did not reach the trace: {e}")
            self.assertIs(e.get("verified"), True,
                          f"a passing run was not marked verified: {e}")

    def test_a_failing_run_is_distinguishable_from_a_passing_one(self):
        with TemporaryDirectory() as tmp:
            runs = _run(tmp, ["test_probe.py"],
                        body="def test_no():\n    assert False\n")
            self.assertEqual(len(runs), 1)
            e = runs[0]
            self.assertNotEqual(e.get("returncode"), 0,
                                f"a FAILING suite reported returncode 0: {e}")
            self.assertIs(e.get("verified"), False,
                          f"a failing run was marked verified: {e}")

    def test_ok_alone_cannot_separate_them(self):
        # The reason a new field was needed. `ok` means the runner ran.
        with TemporaryDirectory() as tmp:
            good = _run(tmp, ["test_probe.py"])[0]
        with TemporaryDirectory() as tmp:
            bad = _run(tmp, ["test_probe.py"],
                       body="def test_no():\n    assert False\n")[0]
        self.assertEqual(good["ok"], bad["ok"],
                         "if `ok` ever separated pass from fail this field would "
                         "be unnecessary -- but it does not, and cycle 79 is why")
        self.assertNotEqual(good.get("returncode"), bad.get("returncode"))


if __name__ == "__main__":
    unittest.main()
