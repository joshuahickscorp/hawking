#!/usr/bin/env python3.12
"""Tests for HIDE YOU research controller (Y5).

Locks the five properties that matter:

  1. Four-way separation by type; category change requires a recorded transition
  2. Every factual claim links to evidence or is explicitly UNSUPPORTED
  3. Research runs leave durable memory untouched (promotion is explicit)
  4. Contradictions surface both sides; no auto-pick by grade or recency
  5. Freshness: source captured-at + claim staleness at answer time

Plus graph relation: refuting a source undermines claims that rest on it
via ResearchObjectGraph propagation.

    python3.12 -m ramanujan.test_you_research
"""
from __future__ import annotations

import unittest
from dataclasses import replace

from ramanujan.you_research import (
    AUTHORITY,
    Claim,
    ClaimCategory,
    ClaimEvidenceGraph,
    Contradiction,
    DurableMemory,
    EvidenceBinding,
    EvidenceRequired,
    FakeRetriever,
    PromotionRefused,
    ResearchController,
    ResearchMode,
    TransitionRefused,
    freshness_for_claim,
    load_fixture_documents,
    load_seed_claims,
    offline_controller,
)


class TestFourWaySeparation(unittest.TestCase):
    """Property 1: four categories; no silent category change."""

    def test_four_categories_exist(self) -> None:
        names = {c.value for c in ClaimCategory}
        self.assertEqual(
            names,
            {
                "retrieved_fact",
                "model_inference",
                "user_provided",
                "uncertain_hypothesis",
            },
        )

    def test_claim_is_frozen_category_cannot_be_assigned(self) -> None:
        c = Claim(
            id="c1",
            text="x",
            category=ClaimCategory.MODEL_INFERENCE,
            evidence=EvidenceBinding.UNSUPPORTED,
            citation_ids=(),
            subject=None,
            value=None,
            confidence=0.2,
            created_at="2024-06-01T00:00:00Z",
        )
        with self.assertRaises(Exception):
            c.category = ClaimCategory.RETRIEVED_FACT  # type: ignore[misc]

    def test_transition_requires_reason_and_is_recorded(self) -> None:
        g = ClaimEvidenceGraph()
        claim = Claim(
            id="c_inf",
            text="maybe 120 us",
            category=ClaimCategory.UNCERTAIN_HYPOTHESIS,
            evidence=EvidenceBinding.UNSUPPORTED,
            citation_ids=(),
            subject="qubit_coherence_time_us",
            value="120",
            confidence=0.1,
            created_at="2024-06-01T00:00:00Z",
        )
        g.add_claim(claim)
        with self.assertRaises(TransitionRefused):
            g.transition_category("c_inf", ClaimCategory.RETRIEVED_FACT, reason="", actor="test")
        new = g.transition_category(
            "c_inf",
            ClaimCategory.MODEL_INFERENCE,
            reason="reclassified as model synthesis, still not a retrieved fact",
            actor="controller",
        )
        self.assertEqual(new.category, ClaimCategory.MODEL_INFERENCE)
        self.assertEqual(len(g.transitions), 1)
        self.assertEqual(g.transitions[0].from_category, ClaimCategory.UNCERTAIN_HYPOTHESIS)
        self.assertEqual(g.transitions[0].to_category, ClaimCategory.MODEL_INFERENCE)
        self.assertEqual(g.transitions[0].reason.startswith("reclassified"), True)

    def test_inference_cannot_silently_become_fact(self) -> None:
        """The failure this exists to prevent: inference → fact without a transition."""
        g = ClaimEvidenceGraph()
        claim = Claim(
            id="c_inf2",
            text="inferred capital is Port Aster",
            category=ClaimCategory.MODEL_INFERENCE,
            evidence=EvidenceBinding.UNSUPPORTED,
            citation_ids=(),
            subject="synthetic_polity_capital",
            value="Port Aster",
            confidence=0.4,
            created_at="2024-06-01T00:00:00Z",
        )
        g.add_claim(claim)
        # No path that mutates category without transition_category
        self.assertEqual(g.claims["c_inf2"].category, ClaimCategory.MODEL_INFERENCE)
        # Forging a fact would require an explicit transition (and still evidence law)
        with self.assertRaises(TransitionRefused):
            g.transition_category(
                "c_inf2",
                ClaimCategory.RETRIEVED_FACT,
                reason="",  # empty reason refused
                actor="sneaky",
            )
        # Even with a reason, becoming RETRIEVED_FACT while UNSUPPORTED+no citations
        # is still a MODEL→FACT transition that is *recorded* — caller must also
        # satisfy evidence law if they later mark it factual with no evidence.
        # RETRIEVED_FACT is factual, so after transition assert_evidence_law runs.
        # UNSUPPORTED with empty citations is legal for factual claims.
        promoted = g.transition_category(
            "c_inf2",
            ClaimCategory.RETRIEVED_FACT,
            reason="explicit promotion after source attach (still unsupported until linked)",
            actor="auditor",
        )
        self.assertEqual(promoted.category, ClaimCategory.RETRIEVED_FACT)
        self.assertEqual(promoted.evidence, EvidenceBinding.UNSUPPORTED)
        self.assertEqual(len(g.transitions), 1)


class TestEvidenceLaw(unittest.TestCase):
    """Property 2: every factual claim is LINKED or UNSUPPORTED. No third state."""

    def test_linked_requires_citations(self) -> None:
        bad = Claim(
            id="bad",
            text="something",
            category=ClaimCategory.RETRIEVED_FACT,
            evidence=EvidenceBinding.LINKED,
            citation_ids=(),
            subject=None,
            value=None,
            confidence=0.5,
            created_at="2024-06-01T00:00:00Z",
        )
        with self.assertRaises(EvidenceRequired):
            bad.assert_evidence_law()

    def test_unsupported_forbids_citations(self) -> None:
        bad = Claim(
            id="bad2",
            text="something",
            category=ClaimCategory.USER_PROVIDED,
            evidence=EvidenceBinding.UNSUPPORTED,
            citation_ids=("cit1",),
            subject=None,
            value=None,
            confidence=1.0,
            created_at="2024-06-01T00:00:00Z",
        )
        with self.assertRaises(EvidenceRequired):
            bad.assert_evidence_law()

    def test_only_two_evidence_states(self) -> None:
        self.assertEqual(
            {e.value for e in EvidenceBinding},
            {"linked", "unsupported"},
        )

    def test_run_claims_obey_evidence_law(self) -> None:
        ctl = offline_controller()
        result = ctl.run(
            "qubit coherence time",
            mode=ResearchMode.CITED_ANSWER,
            answer_at="2024-09-01T00:00:00Z",
        )
        for c in result.claims:
            c.assert_evidence_law()
            if c.category is ClaimCategory.RETRIEVED_FACT:
                self.assertEqual(c.evidence, EvidenceBinding.LINKED)
                self.assertTrue(c.citation_ids)


class TestDurableMemoryUntouched(unittest.TestCase):
    """Property 3: research run leaves durable memory untouched."""

    def test_run_does_not_write_durable_memory(self) -> None:
        mem = DurableMemory()
        mem.entries["preexisting"] = {"claim_id": "preexisting", "text": "user pref"}
        mem.write_count = 1
        before_fp = mem.fingerprint()
        before_writes = mem.write_count

        ctl = ResearchController(
            retriever=FakeRetriever(),
            durable_memory=mem,
            seed_claims=load_seed_claims(),
        )
        result = ctl.run(
            "qubit coherence",
            mode=ResearchMode.DEEP_RESEARCH,
            answer_at="2024-09-01T00:00:00Z",
        )
        self.assertTrue(result.durable_memory_untouched)
        self.assertEqual(mem.fingerprint(), before_fp)
        self.assertEqual(mem.write_count, before_writes)
        self.assertIn("preexisting", mem.entries)
        # retrieved claims exist in the run but not in durable memory
        retrieved = [c for c in result.claims if c.category is ClaimCategory.RETRIEVED_FACT]
        self.assertGreater(len(retrieved), 0)
        for c in retrieved:
            self.assertNotIn(c.id, mem.entries)

    def test_promotion_is_explicit_and_recorded(self) -> None:
        mem = DurableMemory()
        ctl = ResearchController(
            retriever=FakeRetriever(),
            durable_memory=mem,
            seed_claims=load_seed_claims(),
        )
        result = ctl.run("Ramsey interferometry", mode=ResearchMode.QUICK_SEARCH)
        retrieved = [c for c in result.claims if c.category is ClaimCategory.RETRIEVED_FACT]
        self.assertTrue(retrieved)
        target = retrieved[0]

        with self.assertRaises(PromotionRefused):
            ctl.promote_to_memory(target.id, actor="x", reason="")

        promo = ctl.promote_to_memory(
            target.id,
            actor="user",
            reason="explicitly keep this methods claim",
        )
        self.assertEqual(promo.claim_id, target.id)
        self.assertIn(target.id, mem.entries)
        self.assertEqual(mem.entries[target.id]["reason"], "explicitly keep this methods claim")
        self.assertEqual(len(mem.promotions), 1)


class TestContradictionSurfacesBoth(unittest.TestCase):
    """Property 4: disagreement → recorded contradiction, both sides, no auto-pick."""

    def test_coherence_contradiction_not_auto_resolved(self) -> None:
        ctl = offline_controller()
        # Broad query retrieves both Alpha (120) and Beta (80)
        result = ctl.run(
            "qubit coherence time microseconds",
            mode=ResearchMode.FACT_AUDIT,
            k=10,
            answer_at="2024-09-01T00:00:00Z",
        )
        coh = [
            c
            for c in result.claims
            if c.subject == "qubit_coherence_time_us"
            and c.category is ClaimCategory.RETRIEVED_FACT
        ]
        values = {c.value for c in coh}
        self.assertIn("120", values)
        self.assertIn("80", values)

        # At least one contradiction between differing values
        subj_contra = [
            x for x in result.contradictions if x.subject == "qubit_coherence_time_us"
        ]
        self.assertGreater(len(subj_contra), 0)
        for x in subj_contra:
            self.assertEqual(x.resolution, "unresolved_both_surfaced")
            self.assertIsNone(x.preferred_claim_id)
            self.assertNotEqual(x.value_a, x.value_b)
            # Both claim ids remain present in the result — neither dropped
            ids = {c.id for c in result.claims}
            self.assertIn(x.claim_a_id, ids)
            self.assertIn(x.claim_b_id, ids)

    def test_higher_grade_source_is_not_preferred(self) -> None:
        """Controller must not resolve by preferring the higher-graded source."""
        ctl = offline_controller()
        result = ctl.run(
            "qubit coherence",
            mode=ResearchMode.COMPARISON,
            k=10,
            answer_at="2024-09-01T00:00:00Z",
        )
        grades = {g["source_id"]: g["grade"] for g in result.source_grades}
        # Alpha is higher authority than press, but contradictions still unresolved
        for x in result.contradictions:
            self.assertIsNone(x.preferred_claim_id)
            # Neither side auto-selected as winner based on grade
            ga = grades.get(x.source_a_id)
            gb = grades.get(x.source_b_id)
            if ga is not None and gb is not None and ga != gb:
                # Explicitly still unresolved despite grade gap
                self.assertEqual(x.resolution, "unresolved_both_surfaced")

    def test_newer_source_is_not_preferred(self) -> None:
        ctl = offline_controller()
        result = ctl.run(
            "synthetic test polity capital",
            mode=ResearchMode.COMPARISON,
            k=10,
            answer_at="2024-09-01T00:00:00Z",
        )
        capital = [x for x in result.contradictions if x.subject == "synthetic_polity_capital"]
        self.assertGreater(len(capital), 0)
        for x in capital:
            self.assertIsNone(x.preferred_claim_id)
            self.assertEqual({x.value_a, x.value_b}, {"Port Aster", "Harbor Gate"})

    def test_record_contradiction_refuses_preference(self) -> None:
        g = ClaimEvidenceGraph()
        with self.assertRaises(ValueError):
            g.record_contradiction(
                Contradiction(
                    id="x",
                    subject="s",
                    claim_a_id="a",
                    claim_b_id="b",
                    source_a_id="sa",
                    source_b_id="sb",
                    value_a="1",
                    value_b="2",
                    preferred_claim_id="a",  # illegal
                )
            )


class TestFreshness(unittest.TestCase):
    """Property 5: source captured-at + claim staleness at answer time."""

    def test_every_retrieved_fact_carries_captured_at(self) -> None:
        ctl = offline_controller()
        result = ctl.run(
            "qubit coherence",
            mode=ResearchMode.CITED_ANSWER,
            answer_at="2024-09-01T00:00:00Z",
        )
        for c in result.claims:
            if c.category is ClaimCategory.RETRIEVED_FACT:
                self.assertIsNotNone(c.source_captured_at)
                self.assertTrue(c.source_captured_at.endswith("Z") or "+" in c.source_captured_at)

    def test_staleness_computed_at_answer_time(self) -> None:
        docs = load_fixture_documents()
        alpha = next(d for d in docs if d.id == "doc_alpha_2024")
        # Build a minimal claim with alpha's captured_at
        claim = Claim(
            id="f1",
            text="t",
            category=ClaimCategory.RETRIEVED_FACT,
            evidence=EvidenceBinding.UNSUPPORTED,
            citation_ids=(),
            subject=None,
            value=None,
            confidence=0.5,
            created_at="2024-06-01T00:00:00Z",
            source_captured_at=alpha.captured_at,  # 2024-06-01
        )
        # Answer shortly after capture → not stale under 90-day threshold
        fresh = freshness_for_claim(
            claim, "2024-06-15T00:00:00Z", threshold_seconds=90 * 24 * 3600
        )
        self.assertFalse(fresh.stale)
        self.assertIsNotNone(fresh.age_seconds)
        self.assertEqual(fresh.source_captured_at, alpha.captured_at)
        self.assertEqual(fresh.answer_at, "2024-06-15T00:00:00Z")

        # Answer a year later → stale
        stale = freshness_for_claim(
            claim, "2025-06-15T00:00:00Z", threshold_seconds=90 * 24 * 3600
        )
        self.assertTrue(stale.stale)
        self.assertGreater(stale.age_seconds or 0, 90 * 24 * 3600)

    def test_run_freshness_reports_match_claims(self) -> None:
        ctl = offline_controller()
        answer_at = "2025-01-01T00:00:00Z"
        result = ctl.run(
            "qubit coherence",
            mode=ResearchMode.DEEP_RESEARCH,
            answer_at=answer_at,
            freshness_threshold_seconds=30 * 24 * 3600,  # 30 days
        )
        self.assertTrue(result.freshness)
        for fr in result.freshness:
            self.assertEqual(fr.answer_at, answer_at)
            self.assertIsNotNone(fr.source_captured_at)
            # All fixture docs captured 2024-06-01 → stale by 2025-01-01 at 30d
            self.assertTrue(fr.stale)


class TestClaimEvidenceGraphRelation(unittest.TestCase):
    """Claim–evidence graph uses ResearchObjectGraph; refute propagates."""

    def test_refuting_source_undermines_claims(self) -> None:
        ctl = offline_controller()
        result = ctl.run(
            "qubit coherence time",
            mode=ResearchMode.CITED_ANSWER,
            k=10,
            answer_at="2024-09-01T00:00:00Z",
            add_unsupported_inference=False,
        )
        # Pick a retrieved fact and refute its source
        fact = next(c for c in result.claims if c.category is ClaimCategory.RETRIEVED_FACT)
        source_id = fact.origin_source_id
        self.assertIsNotNone(source_id)
        undermined = ctl.graph.refute_source(source_id, "source withdrawn")
        # Claim depends on citation depends on doc depends on source → claim undermined
        self.assertIn(fact.id, undermined)
        standing = ctl.graph.standing()
        self.assertIn(source_id, standing["refuted"])
        self.assertIn(fact.id, standing["undermined"])

    def test_graph_is_research_object_graph(self) -> None:
        g = ClaimEvidenceGraph()
        from ramanujan.cognition import ResearchObjectGraph

        self.assertIsInstance(g.rog, ResearchObjectGraph)


class TestModesAndOffline(unittest.TestCase):
    def test_all_modes_run_offline(self) -> None:
        ctl = offline_controller()
        for mode in ResearchMode:
            result = ctl.run(
                "coherence methods Ramsey",
                mode=mode,
                k=5,
                answer_at="2024-09-01T00:00:00Z",
            )
            self.assertEqual(result.authority, AUTHORITY)
            self.assertTrue(result.durable_memory_untouched)
            self.assertTrue(result.checkpoints)
            self.assertEqual(result.checkpoints[0].kind, "opened")
            self.assertEqual(result.checkpoints[-1].kind, "complete")

    def test_fixture_corpus_is_committed_and_loadable(self) -> None:
        docs = load_fixture_documents()
        self.assertGreaterEqual(len(docs), 5)
        seeds = load_seed_claims()
        self.assertGreaterEqual(len(seeds), 5)
        # No network: FakeRetriever only sees fixture docs
        r = FakeRetriever(docs)
        hits = r.retrieve("coherence Ramsey", k=3)
        self.assertTrue(all(h.id.startswith("doc_") for h in hits))

    def test_user_provided_claim(self) -> None:
        ctl = offline_controller()
        c = ctl.add_user_claim("I measured 100 us myself", subject="qubit_coherence_time_us", value="100")
        self.assertEqual(c.category, ClaimCategory.USER_PROVIDED)
        self.assertEqual(c.evidence, EvidenceBinding.UNSUPPORTED)
        c.assert_evidence_law()

    def test_cited_answer_surface(self) -> None:
        ctl = offline_controller()
        result = ctl.run("coherence", mode=ResearchMode.CITED_ANSWER, k=10)
        surface = result.cited_answer()
        self.assertIn("claims", surface)
        self.assertIn("citations", surface)
        self.assertIn("contradictions", surface)
        self.assertIn("freshness", surface)
        self.assertTrue(surface["durable_memory_untouched"])


class TestNoSilentFactFromInferenceViaReplace(unittest.TestCase):
    """replace() can forge a new Claim object, but the graph only accepts transition_category."""

    def test_graph_does_not_auto_accept_forged_category(self) -> None:
        g = ClaimEvidenceGraph()
        claim = Claim(
            id="c",
            text="t",
            category=ClaimCategory.MODEL_INFERENCE,
            evidence=EvidenceBinding.UNSUPPORTED,
            citation_ids=(),
            subject=None,
            value=None,
            confidence=0.3,
            created_at="2024-06-01T00:00:00Z",
        )
        g.add_claim(claim)
        forged = replace(claim, category=ClaimCategory.RETRIEVED_FACT)
        # Forged object exists in Python land, but graph still holds the original
        self.assertEqual(g.claims["c"].category, ClaimCategory.MODEL_INFERENCE)
        self.assertEqual(forged.category, ClaimCategory.RETRIEVED_FACT)
        # Graph category only changes through transition_category
        self.assertEqual(len(g.transitions), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
