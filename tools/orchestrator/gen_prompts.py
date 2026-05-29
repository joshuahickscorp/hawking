#!/usr/bin/env python3
"""gen_prompts — emit a diverse newline-delimited prompt corpus for local
quantized-residual capture.

Diversity (not literary quality) is what matters: we want the captured
residuals to span the token distribution the head will face at inference —
code, prose, reasoning, enumeration, Q&A, summarization. Templates × topics
give a few hundred varied prompts deterministically.
"""
from __future__ import annotations

import argparse
import random

TOPICS = [
    "photosynthesis", "the French Revolution", "quantum entanglement",
    "binary search trees", "the stock market", "machine learning",
    "the water cycle", "black holes", "the immune system", "blockchain",
    "climate change", "the Roman Empire", "neural networks", "DNA replication",
    "the internet", "volcanoes", "compound interest", "the human brain",
    "renewable energy", "the printing press", "antibiotics", "tectonic plates",
    "supply and demand", "the Cold War", "genetic engineering", "tides",
    "the electoral college", "vaccines", "cryptography", "the Big Bang",
    "coral reefs", "inflation", "the nervous system", "solar panels",
    "the Silk Road", "recursion", "evolution", "the greenhouse effect",
    "democracy", "the speed of light", "fermentation", "gravity",
    "the stock exchange", "photosynthetic bacteria", "the Industrial Revolution",
    "operating systems", "the periodic table", "ocean currents", "memory in computers",
    "the theory of relativity",
]

LANGS = ["Rust", "Python", "Go", "C", "JavaScript", "Haskell"]
ALGOS = [
    "a binary search tree", "quicksort", "a hash map", "Dijkstra's algorithm",
    "a linked list", "merge sort", "a bloom filter", "an LRU cache",
    "depth-first search", "a trie", "matrix multiplication", "a ring buffer",
]
PLACES = ["a lighthouse", "an old library", "a space station", "a mountain village",
          "a submarine", "a desert outpost", "a train station", "a clockmaker's shop"]
TEMPLATES = [
    "Explain in detail how {topic} works.",
    "Describe the key concepts behind {topic} for a beginner.",
    "What are the main causes and effects of {topic}?",
    "Summarize {topic} in a few clear paragraphs.",
    "Compare and contrast two perspectives on {topic}.",
    "Write a short story set in {place}.",
    "Describe a character who lives near {place}.",
    "Implement {algo} in {lang}, with commented code.",
    "Write a {lang} function that demonstrates {algo} and explain it.",
    "List five practical tips related to {topic}.",
    "Give step-by-step instructions to understand {topic}.",
    "Why is {topic} important, and what are common misconceptions?",
    "Draft an email explaining {topic} to a colleague.",
    "Pose three thoughtful questions about {topic} and answer them.",
    "Explain {topic} using an everyday analogy.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    seen = set()
    out = []
    attempts = 0
    while len(out) < args.n and attempts < args.n * 20:
        attempts += 1
        t = rng.choice(TEMPLATES)
        p = t.format(
            topic=rng.choice(TOPICS),
            place=rng.choice(PLACES),
            algo=rng.choice(ALGOS),
            lang=rng.choice(LANGS),
        )
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"[gen_prompts] wrote {len(out)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
