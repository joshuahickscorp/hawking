"""The builder door must advertise the ops its applier actually implements.

Measured, cycles 69-74 of the self-development campaign: the body reached for
`repo.edit` five times running and lost every one to the same failure -- its
`operations` payload truncated mid-JSON, because it had chosen `create` and was
emitting a whole file inline. The engine's applier supports `append`, which
needs no anchor and no file body and is the smallest complete mutation that
exists; `BUILDER_SHAPES` never mentioned it. `insert_before` and `replace_file`
were missing too.

A contract that names three of six ops is not merely incomplete. It steers a
budget-limited body toward the most expensive op in the set, and the generation
window is the binding constraint: 5632-token context, ~2534 tokens left for
generation after the seed and the tool contract. A whole inline file does not
fit; an append does.

So this pins agreement between two things that had drifted:

  * every op `_apply_operations` implements is named in the advertised shape
  * the shape names no op the applier would reject

and it proves the cheap op actually works end to end through the builder door,
rather than trusting the string.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.chat_tools import BUILDER_SHAPES, run_builder_tool
from hcli.engine import Engine
from hcli.workspace import Workspace

#: The ops `_apply_operations` dispatches on. Read from the source rather than
#: retyped, so an op added to the applier fails this test until it is taught.
_APPLIER = Path(__file__).resolve().parents[1] / "engine.py"


def _implemented_ops() -> set:
    src = _APPLIER.read_text(encoding="utf-8")
    start = src.index("def _apply_operations")
    end = src.index("unsupported mutation op", start)
    return set(re.findall(r'op == "([a-z_]+)"', src[start:end]))


def _advertised_ops() -> set:
    shape = BUILDER_SHAPES["repo.edit"]
    match = re.search(r'"op": "([a-z_|]+)"', shape)
    return set(match.group(1).split("|")) if match else set()


class _Pool:
    model_path = "sealed-3.14"
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


class TestTheMenuMatchesTheApplier(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.engine = Engine(Workspace(str(self.root)),
                             runtime_provider=lambda: _Pool())

    def test_the_applier_implements_more_than_one_op(self):
        # Guards the reader above: if the source shape ever changes so the regex
        # matches nothing, every other assertion here passes vacuously.
        self.assertGreater(len(_implemented_ops()), 1,
                           "the op reader found nothing -- the assertions below "
                           "would pass against an empty set")

    def test_every_implemented_op_is_advertised(self):
        missing = _implemented_ops() - _advertised_ops()
        self.assertEqual(missing, set(),
                         f"the applier implements {sorted(missing)} and the "
                         f"builder contract never says so, so a body that must "
                         f"fit its call in a generation window cannot choose them")

    def test_no_advertised_op_would_be_refused(self):
        extra = _advertised_ops() - _implemented_ops()
        self.assertEqual(extra, set(),
                         f"the contract offers {sorted(extra)}, which the "
                         f"applier rejects as unsupported")

    def test_append_is_a_complete_mutation_with_no_anchor_and_no_file_body(self):
        # The whole point of naming `append`: this is the smallest payload that
        # changes a file, so it is the one a truncated generation can finish.
        (self.root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
        result = run_builder_tool("repo.edit", {"operations": [
            {"op": "append", "path": "mod.py", "new_lines": ["OTHER = 2"]}]},
            engine=self.engine)
        self.assertTrue(result.ok, getattr(result, "error", None))
        self.assertIn(result.value["status"], ("accepted", "unproven"),
                      f"append was refused: {result.value}")
        self.assertEqual((self.root / "mod.py").read_text(),
                         "VALUE = 1\nOTHER = 2\n")


if __name__ == "__main__":
    unittest.main()
