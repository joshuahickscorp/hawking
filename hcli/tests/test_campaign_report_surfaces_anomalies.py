"""The campaign report must surface the defects that were found by hand.

Every anomaly check here corresponds to something that actually reached a receipt
before anyone noticed, so this is a regression test against real history rather
than a demonstration of a formatter.

The two that cost the most:

  * cycle 74 logged "repo.edit CALLED and accepted" against a clean worktree and
    an unmoved HEAD, because `ok` is True for a REJECTED mutation by design;
  * cycle 79 labelled a test that exited 0 as RED, and the phase machine advanced
    on it. The body self-corrected; the false-positive class did not.

If this file passes while those go unreported, the report is decoration.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hcli.campaign_report import Cycle, _find_anomalies, arcs, gather, render


def _anoms(note, trace):
    c = Cycle(n=1, note=note, trace=trace)
    _find_anomalies({1: c})
    return c.anomalies


class TestTheReportSurfacesWhatTheLogHid(unittest.TestCase):
    def test_a_red_on_a_passing_test_is_reported(self):
        got = _anoms("RED: hcli/tests/test_x.py — 5 passed in 0.06s (returncode 0)",
                     [{"tool": "tests.run", "ok": True, "returncode": 0}])
        self.assertTrue(any("PASSED" in a for a in got),
                        f"a green discriminator labelled RED went unreported: {got}")

    def test_a_real_red_is_not_reported_as_an_anomaly(self):
        # The negative control. If every RED trips this, the check is noise.
        got = _anoms("RED: hcli/tests/test_x.py — 1 failed in 0.900s",
                     [{"tool": "tests.run", "ok": True, "returncode": 1}])
        self.assertFalse(any("PASSED" in a for a in got),
                         f"a genuine RED was flagged as an anomaly: {got}")

    def test_repo_edit_ok_without_a_verdict_is_reported(self):
        got = _anoms("repo.edit CALLED and accepted",
                     [{"tool": "repo.edit", "ok": True}])
        self.assertTrue(any("never that the repository changed" in a for a in got),
                        f"cycle 74's exact shape went unreported: {got}")

    def test_a_rejected_mutation_logged_as_accepted_is_reported(self):
        got = _anoms("repo.edit CALLED and accepted",
                     [{"tool": "repo.edit", "ok": True, "verdict": "rejected",
                       "applied": False}])
        self.assertTrue(any("rejected" in a and "accepted" in a for a in got),
                        f"a rejected mutation logged as accepted went unreported: {got}")

    def test_a_failed_tool_call_is_reported_with_its_error(self):
        got = _anoms("orienting", [{"tool": "fs.read", "ok": False,
                                    "error": "FileNotFoundError: H-MANIFESTO.md does not exist"}])
        self.assertTrue(any("H-MANIFESTO" in a for a in got),
                        f"the missing-manifesto call went unreported: {got}")

    def test_an_accepted_mutation_is_not_flagged(self):
        got = _anoms("repo.edit APPLIED to the tree (verdict: unproven)",
                     [{"tool": "repo.edit", "ok": True, "verdict": "unproven",
                       "applied": True}])
        self.assertEqual(got, [], f"a genuine applied mutation was flagged: {got}")


class TestItReadsARealCampaignTree(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        sd = self.root / ".hcli" / "selfdev"
        (sd / "evidence").mkdir(parents=True)
        (sd / "campaign.log").write_text(
            "[10:00:00] cycle 1: reachable 11/11, HEAD abc1234 | "
            "ORIENT->DISCRIMINATOR_REQUIRED (hypothesis stated: x is broken) | DID: HYPOTHESIS: x\n"
            "[10:10:00] cycle 2: reachable 11/11, HEAD abc1234 | "
            "DISCRIMINATOR_REQUIRED->ACT_REQUIRED (RED EXECUTED via tests.run) | DID: RED: 5 passed\n"
            "[10:20:00] cycle 3 soft-terminal (TimeoutError: timed out); backing off. total 1\n"
            "[10:30:00] campaign resumed at cycle 3, HEAD abc1234, reachable 11/11\n")
        (sd / "state.json").write_text(json.dumps(
            {"cycle": 3, "phase_name": "ACT_REQUIRED", "accepted_changes": 0,
             "hypothesis": "x is broken", "failed_cycles": 0}))
        (sd / "evidence" / "cycle-0002.trace.json").write_text(json.dumps(
            [{"tool": "tests.run", "ok": True, "returncode": 0}]))

    def test_it_parses_cycles_softs_and_restarts(self):
        d = gather(self.root)
        self.assertEqual(sorted(d["cycles"]), [1, 2])
        self.assertEqual(len(d["softs"]), 1)
        self.assertEqual(len(d["resumes"]), 1)
        self.assertEqual(d["cycles"][1].reachable, "11/11",
                         "reachability must not swallow the trailing comma")

    def test_it_groups_cycles_into_an_arc(self):
        d = gather(self.root)
        A = arcs(d["cycles"])
        self.assertEqual(len(A), 1)
        self.assertEqual([c.n for c in A[0]["cycles"]], [1, 2])

    def test_the_rendered_report_names_the_anomaly(self):
        out = render(gather(self.root))
        self.assertIn("PASSED", out,
                      "the report rendered without the anomaly it exists to surface")
        self.assertIn("soft-terminals", out)
        self.assertIn("cycle 3", out)


if __name__ == "__main__":
    unittest.main()
