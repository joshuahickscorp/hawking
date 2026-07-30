"""Paths and Mathlib pin for local Ramanujan data generation."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PKG = Path(__file__).resolve().parent
CORPORA = DATA_PKG / "corpora"

# Pinned Mathlib (read-only). Override with RAMANUJAN_MATHLIB for tests.
_DEFAULT_MATHLIB = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Hawking"
    / "Ramanujan"
    / "mathlib4"
)
MATHLIB_ROOT = Path(os.environ.get("RAMANUJAN_MATHLIB", str(_DEFAULT_MATHLIB)))

# Recorded in environment lock; re-read from git when available.
EXPECTED_MATHLIB_COMMIT = "2ec0166b31100827cd34bacca4d3b9ea3da9d618"
EXPECTED_LEAN_VERSION = "4.32.1"

# Bounded modules for the first real corpus (scale-up widens this list).
DEFAULT_MODULES = [
    "Mathlib/Data/Nat/Basic.lean",
    "Mathlib/Data/Nat/Order/Basic.lean",
    "Mathlib/Data/Nat/Order/Lemmas.lean",
    "Mathlib/Data/Nat/Bitwise.lean",
    "Mathlib/Data/Nat/Sqrt.lean",
    "Mathlib/Data/Nat/GCD/Basic.lean",
    "Mathlib/Data/Nat/Prime/Basic.lean",
    "Mathlib/Data/List/Basic.lean",
    "Mathlib/Data/List/Length.lean",
    "Mathlib/Logic/Basic.lean",
    "Mathlib/Logic/Function/Basic.lean",
    "Mathlib/Algebra/Group/Basic.lean",
    "Mathlib/Algebra/Group/Defs.lean",
    "Mathlib/Algebra/Ring/Basic.lean",
    "Mathlib/Order/Basic.lean",
    "Mathlib/Init.lean",
]

COUNTEREXAMPLE_DIR = "Counterexamples"

GENERATION_RECEIPT = CORPORA / "GENERATION_RECEIPT.json"
CONTAMINATION_RECEIPT = CORPORA / "CONTAMINATION_RECEIPT.json"

SOURCE_FILES = {
    "D1": CORPORA / "d1_proof_traces.jsonl",
    "D2": CORPORA / "d2_state_transitions.jsonl",
    "D3": CORPORA / "d3_premise_pairs.jsonl",
    "D4": CORPORA / "d4_repair_pairs.jsonl",
    "D6": CORPORA / "d6_counterexamples.jsonl",
    "D7": CORPORA / "d7_tool_use_traces.jsonl",
}
