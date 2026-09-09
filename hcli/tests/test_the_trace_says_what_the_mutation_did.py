"""A mutation's trace entry must carry its VERDICT, not just that it ran.

Measured, campaign cycle 74. The supervisor logged

    ACT_REQUIRED->VERIFY (repo.edit CALLED and accepted)

and nothing had been written. No file changed in the worktree, no file changed
in the primary checkout, HEAD did not move. The body had reached `repo.edit`,
the transaction had run, and the mutation had NOT been accepted.

The cause is a deliberate design decision one layer down, read wrongly one layer
up. `run_builder_tool` returns `ok=True` for a REJECTED mutation on purpose --
"a rejected mutation is NOT an error the model should paper over: it is a result
it must report" -- so `ok` means the call was well-formed and the transaction
completed. It has never meant the repository changed. But the trace entry
`run_with_tools` appends carries only `tool`, `arguments`, `ok`, `error` and
`provenance`, and of those only `ok` is a documented, cross-tool field.

The verdict was not quite absent -- `repo.edit` happens to put `status` in its
`provenance`, which is a free-form per-tool dict every other tool fills with
something else entirely (a source URL, a retrieval time). Nothing in the trace's
contract says a mutation puts its verdict there, so no consumer could rely on
it, and the one field every consumer CAN rely on says the wrong thing. A truth
reachable only by knowing one tool's private dict shape is not an interface.

So every consumer of the trace -- a supervisor, a receipt, an autonomy gate --
is STRUCTURALLY UNABLE to tell a landing from a refusal. That is not a reading
error to fix in one consumer. It is a missing field, and the false positive it
produced is exactly the class that must remain impossible: a HEAD that did not
move and a monitor that says work landed.

These pin that a rejected mutation and an accepted one leave DIFFERENT traces.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.chat_tools import run_with_tools
from hcli.engine import Engine
from hcli.workspace import Workspace


class _Pool:
    model_path = "sealed-3.14"
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


def _replies(*texts):
    """A completer that reads a fixed script, so the test drives the tool loop
    without a resident."""
    script = list(texts)

    def complete(conversation):
        return script.pop(0) if script else "done"
    return complete


def _call(path, op, **extra):
    import json
    payload = {"tool": "repo.edit",
               "arguments": {"operations": [dict(op=op, path=path, **extra)]}}
    return json.dumps(payload)


class TestTheTraceCarriesTheVerdict(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.engine = Engine(Workspace(str(self.root)),
                             runtime_provider=lambda: _Pool())

    def _run(self, reply):
        _, trace = run_with_tools(_replies(reply), [], None, engine=self.engine)
        edits = [e for e in trace if e["tool"] == "repo.edit"]
        self.assertEqual(len(edits), 1, f"expected one repo.edit entry, got {trace}")
        return edits[0]

    def test_an_accepted_mutation_is_recorded_as_such(self):
        entry = self._run(_call("made.py", "create", new_lines=["VALUE = 1"]))
        self.assertTrue(entry["ok"])
        self.assertIn(entry.get("verdict"), ("accepted", "unproven"),
                      f"no usable verdict on an applied mutation: {entry}")
        self.assertTrue(entry.get("applied"), f"applied flag missing: {entry}")
        self.assertEqual((self.root / "made.py").read_text().strip(), "VALUE = 1")

    def test_a_rejected_mutation_does_not_look_like_an_accepted_one(self):
        # A replace whose anchor is not in the file: the transaction runs, the
        # call is well-formed, and nothing is written.
        (self.root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
        entry = self._run(_call("mod.py", "replace",
                                old_lines=["NOT_PRESENT = 9"],
                                new_lines=["VALUE = 2"]))
        self.assertEqual((self.root / "mod.py").read_text(), "VALUE = 1\n",
                         "the file changed despite a rejected mutation")
        self.assertEqual(entry.get("verdict"), "rejected",
                         f"a rejected mutation left no rejected verdict: {entry}")
        self.assertFalse(entry.get("applied"),
                         f"a rejected mutation reported applied: {entry}")

    def test_ok_alone_cannot_separate_them(self):
        # The reason a new field was needed rather than a stricter read of `ok`.
        (self.root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
        good = self._run(_call("made.py", "create", new_lines=["VALUE = 1"]))
        bad = self._run(_call("mod.py", "replace",
                              old_lines=["NOT_PRESENT = 9"], new_lines=["V = 2"]))
        self.assertEqual(good["ok"], bad["ok"],
                         "if `ok` ever separates these two, this whole field is "
                         "unnecessary -- but it does not, and cycle 74 is why")
        self.assertNotEqual(good.get("verdict"), bad.get("verdict"))


if __name__ == "__main__":
    unittest.main()
