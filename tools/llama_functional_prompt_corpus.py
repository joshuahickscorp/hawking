#!/usr/bin/env python3
"""Generate a deterministic local-only Llama student calibration corpus.

The corpus contains prompt templates authored in this repository only.  It is
activation calibration evidence, not a quality benchmark and not a substitute
for an independently held-out generated-token suite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "hawking.tg.llama_functional_prompt_corpus.v1"
TOPICS = ("sorting", "binary search", "hash maps", "HTTP caching", "UTF-8", "floating point", "database indexes", "unit testing")
LANGUAGES = ("Rust", "Python", "TypeScript", "SQL")
SHAPES = ("array", "object", "list", "record")


def prompts(count: int) -> list[str]:
    if count < 2:
        raise ValueError("count must be at least 2")
    rows: list[str] = []
    for index in range(count):
        topic = TOPICS[index % len(TOPICS)]
        language = LANGUAGES[(index // len(TOPICS)) % len(LANGUAGES)]
        shape = SHAPES[(index // (len(TOPICS) * len(LANGUAGES))) % len(SHAPES)]
        a, b = index % 97 + 3, (index * 7) % 89 + 2
        kind = index % 6
        if kind == 0:
            rows.append(f"Case {index}: Explain {topic} in {language} using exactly two concise sentences.")
        elif kind == 1:
            rows.append(f"Case {index}: Write a {language} function that returns {a} plus {b}. Include type annotations.")
        elif kind == 2:
            rows.append(f"Case {index}: Return valid JSON {shape} with keys id={index}, topic={topic!r}, and ok=true.")
        elif kind == 3:
            rows.append(f"Case {index}: Compare {topic} and recursion; state one advantage and one limitation of each.")
        elif kind == 4:
            rows.append(f"Case {index}: Compute ({a} * {b}) - {a} and show a one-line verification.")
        else:
            rows.append(f"Case {index}: Give a short debugging checklist for a {language} program involving {topic}.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    rows = prompts(args.count)
    content = "\n".join(rows) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content)
    receipt = {
        "schema": SCHEMA, "status": "LOCAL_TEMPLATE_CORPUS",
        "prompt_count": len(rows), "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "origin": "deterministic templates in tools/llama_functional_prompt_corpus.py; no external corpus or model completion", "quality_claim": None,
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
