"""A unit whose objective IS the goal must keep it.

The worker packet excises the root goal so an ultragoal is not dumped into
every worker. Correct in general, and wrong for the case that matters most: a
single-obligation mission compiles to one WorkUnit whose description is the
goal, and excising it there removes the entire instruction. The worker received

    OBJECTIVE: obligations=G001 [ROOT_GOAL_OMITTED] Relevant files: hcli/resources.py

and had nothing to do. The model answered exactly as it should have --
"I need to check the goal state to understand what G001 requires" -- and was
recorded as a failed unit.

This is the likeliest explanation for the campaign's earlier degeneracy, where
a mission asked to write a document produced a copy of its own prompt: no
instruction ever reached the worker, so the only text it had to work from was
the scaffolding around the hole where the goal should have been.

MIN_ROOT_EXCISE=80 was the intended guard -- "short goals ARE the unit" -- but
length is the wrong test. A 101-character goal is still the unit.
"""
from __future__ import annotations

import unittest

from hcli.goal import (
    MIN_ROOT_EXCISE,
    ROOT_GOAL_OMITTED,
    _excise_root_goal,
    refuse_goal_dump,
    root_is_the_whole_objective,
)

ROOT = (
    "Read hcli/resources.py and explain in two sentences why pid_is_alive "
    "calls os.waitpid before os.kill."
)


class TestObjectiveKeepsItsInstruction(unittest.TestCase):
    def test_the_live_failure_shape(self):
        self.assertGreater(len(ROOT), MIN_ROOT_EXCISE, "not the case under test")
        description = "obligations=G001 " + ROOT
        self.assertEqual(_excise_root_goal(description, ROOT), description)
        self.assertNotIn(ROOT_GOAL_OMITTED, _excise_root_goal(description, ROOT))

    def test_a_quoted_dump_is_still_excised(self):
        """Negative control: the leak this was built to stop must still be stopped."""
        body = "context " * 40 + ROOT + " trailing " * 40
        self.assertIn(ROOT_GOAL_OMITTED, _excise_root_goal(body, ROOT))
        self.assertNotIn(ROOT, _excise_root_goal(body, ROOT))

    def test_the_predicate_separates_the_two(self):
        self.assertTrue(root_is_the_whole_objective("obligations=G001 " + ROOT, ROOT))
        self.assertFalse(root_is_the_whole_objective("context " * 40 + ROOT, ROOT))
        self.assertFalse(root_is_the_whole_objective("unrelated text", ROOT))

    def test_the_guard_permits_it_on_the_objective_line_only(self):
        packet = f"WORKUNIT: u1\nROLE: implementation\nOBJECTIVE: obligations=G001 {ROOT}\n"
        refuse_goal_dump(packet, ROOT)  # must not raise

    def test_the_guard_still_refuses_it_anywhere_else(self):
        packet = f"WORKUNIT: u1\nOBJECTIVE: do the thing\nSTEERING: [constraint] {ROOT}\n"
        with self.assertRaises(ValueError):
            refuse_goal_dump(packet, ROOT)

    def test_the_guard_refuses_a_mixed_packet(self):
        """One legitimate carrier does not license a second, illegitimate one."""
        packet = (
            f"OBJECTIVE: obligations=G001 {ROOT}\n"
            f"NEIGHBORHOOD: earlier unit said {ROOT}\n"
        )
        with self.assertRaises(ValueError):
            refuse_goal_dump(packet, ROOT)


if __name__ == "__main__":
    unittest.main()
