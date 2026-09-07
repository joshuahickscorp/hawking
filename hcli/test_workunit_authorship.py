"""Who authored this WorkUnit?

The Claude-removal ladder (C1 >=50% HCLI-generated, C2 >=80%, C3/C4 100% over a
window) is a RATIO of WorkUnit authorship. Nothing recorded authorship, so the
ratio could only ever be reconstructed by guesswork after the fact.

The default is deliberately None, not "hcli". Defaulting to the value the goal
is trying to move would manufacture the metric -- every legacy unit would count
as autonomous the moment the field existed. Unrecorded is its own bucket and
the ratio is computed against RECORDED units only, with the unrecorded count
reported beside it so a mostly-blind ledger cannot masquerade as a clean 100%.
"""

from __future__ import annotations

import unittest

from hcli.dag_store import _apply_disk_extras, _unit_to_disk
from hcli.workunit import WorkUnit, authorship_report


class TestWorkUnitAuthorship(unittest.TestCase):
    def test_author_defaults_to_unrecorded_not_hcli(self) -> None:
        wu = WorkUnit(id="u1", role="worker", description="x")
        self.assertIsNone(
            wu.author,
            "defaulting authorship to a value would manufacture the C1 metric",
        )

    def test_author_survives_persistence(self) -> None:
        wu = WorkUnit(id="u1", role="worker", description="x", author="hcli")
        payload = _unit_to_disk(wu)
        self.assertEqual(payload.get("author"), "hcli", "authorship never reached disk")
        restored = WorkUnit(id="u1", role="worker", description="x")
        _apply_disk_extras(restored, payload)
        self.assertEqual(restored.author, "hcli", "authorship lost on reload")

    def test_ratio_is_over_recorded_units_and_names_the_blind_ones(self) -> None:
        units = [
            WorkUnit(id="a", role="worker", description="x", author="hcli"),
            WorkUnit(id="b", role="worker", description="x", author="hcli"),
            WorkUnit(id="c", role="worker", description="x", author="claude"),
            WorkUnit(id="d", role="worker", description="x"),  # unrecorded
        ]
        rep = authorship_report(units)
        self.assertEqual(rep["counts"], {"hcli": 2, "claude": 1})
        self.assertEqual(rep["unrecorded"], 1)
        self.assertEqual(rep["total"], 4)
        self.assertAlmostEqual(rep["hcli_fraction"], 2 / 3, places=6)
        self.assertEqual(rep["hcli_fraction_basis"], "recorded")

    def test_an_all_blind_ledger_refuses_to_report_a_fraction(self) -> None:
        """The negative control. No recorded units must not read as 0% or 100%."""
        units = [WorkUnit(id="a", role="worker", description="x")]
        rep = authorship_report(units)
        self.assertIsNone(
            rep["hcli_fraction"],
            "a fraction computed from zero recorded units is fiction",
        )
        self.assertEqual(rep["hcli_fraction_basis"], "NOT_INSTRUMENTED")


    def test_the_resident_records_itself_as_the_author(self) -> None:
        """The one site where HCLI itself creates work must say so."""
        from hcli.agentos.resident import _child_workunit

        child = _child_workunit(
            "parent-1",
            {"id": "child-1", "description": "do a thing", "dependencies": ["parent-1"]},
        )
        self.assertEqual(
            child.author,
            "hcli",
            "the resident authored this unit but the ledger cannot tell",
        )

    def test_a_repair_inherits_authorship(self) -> None:
        """A repair of an HCLI unit is HCLI's work, not an unrecorded unit."""
        from hcli.workunit import emit_repair

        parent = WorkUnit(id="p1", role="worker", description="x", author="hcli")
        parent.status = "failed"
        rep = emit_repair({"p1": parent}, parent)
        self.assertIsNotNone(rep, "no repair unit was produced")
        self.assertEqual(
            rep.author,
            "hcli",
            "the repair dropped out of the recorded population",
        )


if __name__ == "__main__":
    unittest.main()
