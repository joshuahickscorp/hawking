from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "llama_functional_prompt_corpus.py"
SPEC = importlib.util.spec_from_file_location("llama_functional_prompt_corpus", MODULE_PATH)
assert SPEC and SPEC.loader
corpus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = corpus
SPEC.loader.exec_module(corpus)


def test_templates_are_deterministic_and_nonempty() -> None:
    first, again = corpus.prompts(128), corpus.prompts(128)
    assert first == again
    assert len(first) == 128
    assert len(set(first)) == 128
    assert all("\n" not in row and row for row in first)
