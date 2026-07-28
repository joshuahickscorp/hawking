#!/usr/bin/env python3.12
"""Tests for Mathlib-local Ramanujan data extractors (D1–D4, D6, D7)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ramanujan.data.common import content_hash, dedup_by_hash, stamp_item
from ramanujan.data.extractors import (
    extract_d1,
    extract_d2,
    extract_d3,
    extract_d6,
    extract_d7,
)
from ramanujan.data.parse_mathlib import TheoremDecl, extract_premises, extract_tactics, parse_lean_file
from ramanujan.limits import LimitRegistry


SAMPLE_LEAN = """
namespace Nat

theorem add_comm (n m : ℕ) : n + m = m + n := by
  rw [Nat.add_comm]

lemma mul_one (n : ℕ) : n * 1 = n := by
  simp

theorem fancy (a b : ℕ) : a + b = b + a :=
  Nat.add_comm a b

end Nat
"""


class TestResearchFence(unittest.TestCase):
    def test_research_stays_false(self):
        self.assertFalse(LimitRegistry().research_authorized())


class TestParse(unittest.TestCase):
    def test_parse_sample(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "Basic.lean"
            p.write_text(SAMPLE_LEAN, encoding="utf-8")
            decls = parse_lean_file(p, rel="Mathlib/Data/Nat/Basic.lean")
        names = {d.name for d in decls}
        self.assertIn("add_comm", names)
        self.assertIn("mul_one", names)
        self.assertIn("fancy", names)
        by_name = {d.name: d for d in decls}
        self.assertEqual(by_name["add_comm"].proof_kind, "by")
        self.assertTrue(by_name["add_comm"].tactics)
        self.assertEqual(by_name["fancy"].proof_kind, "term")

    def test_extract_tactics_and_premises(self):
        tacs = extract_tactics("rw [Nat.add_comm]\n  simp", "by")
        self.assertTrue(any(t.startswith("rw") for t in tacs))
        premises = extract_premises("rw [Nat.add_comm, Nat.mul_comm]", "add_comm")
        self.assertTrue(any("add_comm" in p or "Nat.add_comm" in p for p in premises) or premises)


class TestExtractors(unittest.TestCase):
    def _decls(self) -> list[TheoremDecl]:
        return [
            TheoremDecl(
                name="add_comm",
                kind="theorem",
                signature="(n m : ℕ) : n + m = m + n",
                statement="add_comm (n m : ℕ) : n + m = m + n",
                proof="rw [Nat.add_comm]",
                proof_kind="by",
                tactics=["rw [Nat.add_comm]"],
                premises=["Nat.add_comm"],
                file="Mathlib/Data/Nat/Basic.lean",
                line=10,
                module="Mathlib.Data.Nat.Basic",
            ),
            TheoremDecl(
                name="mul_one",
                kind="lemma",
                signature="(n : ℕ) : n * 1 = n",
                statement="mul_one (n : ℕ) : n * 1 = n",
                proof="simp [Nat.mul_one]",
                proof_kind="by",
                tactics=["simp [Nat.mul_one]"],
                premises=["Nat.mul_one"],
                file="Mathlib/Data/Nat/Basic.lean",
                line=20,
                module="Mathlib.Data.Nat.Basic",
            ),
        ]

    def test_d1_d2_d3_shapes(self):
        decls = self._decls()
        d1 = extract_d1(decls, limit=10)
        self.assertGreaterEqual(len(d1), 2)
        self.assertTrue(all(it.get("content_hash") for it in d1))
        self.assertTrue(all(it["provenance"]["RAMANUJAN_RESEARCH_AUTHORIZED"] is False for it in d1))
        self.assertTrue(all(it["provenance"]["teacher_from_math_preserve"] is False for it in d1))

        d2 = extract_d2(decls, limit=10)
        self.assertGreaterEqual(len(d2), 2)
        self.assertTrue(all("tactic" in it and "state_before" in it for it in d2))

        d3 = extract_d3(decls, limit=10)
        self.assertGreaterEqual(len(d3), 1)
        self.assertTrue(all(it.get("positive_premises") for it in d3))

    def test_d6_has_witnesses(self):
        # No Mathlib required for enumerative half — pass a missing root
        with tempfile.TemporaryDirectory() as td:
            items = extract_d6(Path(td), limit=50)
        self.assertGreater(len(items), 10)
        self.assertTrue(all("false_statement" in it or "witness" in it for it in items))
        self.assertTrue(all(it.get("content_hash") for it in items))

    def test_d7_tool_traces(self):
        items = extract_d7(limit=40)
        self.assertGreater(len(items), 5)
        tools = {it["tool"] for it in items}
        self.assertIn("tactic", tools)
        self.assertIn("premise_retrieval", tools)

    def test_stamp_and_dedup(self):
        a = stamp_item({"id": "x", "text": "hello"}, extraction_method="test")
        b = stamp_item({"id": "x", "text": "hello"}, extraction_method="test")
        # content_hash is over body excluding provenance; same body → same hash
        self.assertEqual(a["content_hash"], b["content_hash"])
        deduped = dedup_by_hash([a, b])
        self.assertEqual(len(deduped), 1)

    def test_content_hash_stable(self):
        self.assertEqual(content_hash({"a": 1, "b": 2}), content_hash({"b": 2, "a": 1}))


class TestContaminationHook(unittest.TestCase):
    def test_barrier_admits_formal_math_text(self):
        from tools.odyssey.contamination import build_barrier

        b = build_barrier()
        hits = b.check("theorem add_comm (n m : Nat) : n + m = m + n")
        self.assertEqual(hits, [], msg=hits)

    def test_barrier_rejects_support_halo_leak(self):
        from tools.odyssey.contamination import build_barrier

        b = build_barrier()
        text = (
            'What does BPW stand for in model compression / quantization? '
            'Answer briefly and include the exact phrase "bits per weight".'
        )
        self.assertTrue(b.check(text))


if __name__ == "__main__":
    unittest.main()
