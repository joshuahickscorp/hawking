"""The body reports what it can do -- probed, never asserted.

S036 s3/s4: HCLI perceives "I can inspect source", not "I have ripgrep". S036
s60: the leap from a distribution to an organism is that it KNOWS what it can
do and names what it cannot. So a capability is reachable only when its
underlying primitive answers, and an unowned domain is listed as unowned rather
than pretended into existence.
"""
from __future__ import annotations

import unittest

from hcli.capabilities import (
    UNOWNED_DOMAINS,
    Capability,
    Rung,
    capabilities,
    capability,
    capability_map,
)


class TestSurfaceIsProbed(unittest.TestCase):
    def test_the_owned_domains_are_reachable_against_live_machinery(self):
        m = capability_map()
        # These wrap machinery that exists on HEAD, so they must probe true.
        for domain in ("INSPECT", "RESEARCH", "BUILD", "TEST", "PROCESS", "RECOVER"):
            self.assertIn(domain, m["reachable"], f"{domain} did not probe reachable")

    def test_reachability_is_a_probe_not_an_assertion(self):
        # A capability whose probe returns False is NOT claimed. This is the
        # whole point: the file refuses to advertise a capability that is not
        # wired, the way the reachability census refuses a tool with no caller.
        broken = Capability(
            domain="INSPECT", verb="x",
            ladder=[Rung("a", "b", "cheap")],
            probe=lambda: False)
        self.assertFalse(broken.reachable())
        self.assertFalse(broken.to_dict()["reachable"])

    def test_a_probe_that_raises_is_unreachable_not_a_crash(self):
        def boom():
            raise RuntimeError("primitive gone")
        cap = Capability("X", "x", [Rung("a", "b", "cheap")], probe=boom)
        self.assertFalse(cap.reachable())

    def test_unowned_domains_are_named_not_hidden(self):
        m = capability_map()
        for domain in UNOWNED_DOMAINS:
            self.assertIn(domain, m["not_yet_owned"])
            self.assertNotIn(domain, m["reachable"],
                             f"{domain} is claimed reachable but has no owner")

    def test_debug_is_owned_now_that_it_has_an_implementation(self):
        # DEBUG earned an owner (reproduce + localize). It is claimed only
        # because debug_capability.diagnose exists and probes reachable.
        cap = capability("DEBUG")
        self.assertIsNotNone(cap)
        self.assertTrue(cap.reachable())

    def test_fuzz_is_owned_now_that_it_has_an_implementation(self):
        cap = capability("FUZZ")
        self.assertIsNotNone(cap)
        self.assertTrue(cap.reachable())

    def test_forensics_and_reverse_are_owned_now(self):
        for domain in ("FORENSICS", "REVERSE"):
            cap = capability(domain)
            self.assertIsNotNone(cap, f"{domain} lost its owner")
            self.assertTrue(cap.reachable())

    def test_report_remains_the_last_unowned_domain(self):
        # REPORT has two real paths (receipts + hcli report) but no single owned
        # capability yet; named unowned rather than pretended.
        m = capability_map()
        self.assertEqual(m["not_yet_owned"], ["REPORT"])

    def test_every_capability_has_an_escalation_ladder(self):
        # S036 s11: cheapest sufficient rung first, deeper when evidence demands.
        for cap in capabilities():
            self.assertTrue(cap.ladder, f"{cap.domain} has no ladder")
            costs = [r.cost for r in cap.ladder]
            self.assertTrue(all(c in ("cheap", "moderate", "expensive")
                                for c in costs), f"{cap.domain}: {costs}")

    def test_inspect_ladder_names_its_missing_deeper_rung(self):
        insp = capability("INSPECT")
        deepest = insp.ladder[-1]
        self.assertEqual(deepest.via, "MISSING",
                         "AST/structural search is claimed as owned but is not")

    def test_a_domain_lookup_is_case_insensitive(self):
        self.assertIsNotNone(capability("inspect"))
        self.assertIsNotNone(capability("Inspect"))


if __name__ == "__main__":
    unittest.main()
