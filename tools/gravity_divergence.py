#!/usr/bin/env python3
"""Token-level divergence against the reference model, which is what greedy decoding makes cheap.

The capability gate scores ten deterministic items. That was enough to separate a coherent
artifact from a degenerate one, and it is NOT enough to separate 9/10 from 10/10 -- the
binomial overlap on ten items is enormous, and the over-scale sweep is now sitting on exactly
that distinction.

Greedy decoding is deterministic, so a divergence is a real disagreement rather than sampling
noise, and every generated token is an independent comparison. A hundred prompts at 64 tokens
is 6,400 comparisons instead of 10, at the same GPU cost as a few capability runs.

Reported per prompt: the position of FIRST divergence and whether the sequences re-converge.
Both matter and they mean different things. An artifact that diverges at token 3 and rejoins
is making a lexical choice; one that diverges at token 3 and never rejoins has entered a
different trajectory. Aggregate agreement alone cannot tell those apart, so it is not reported
alone.

The reference is uniform-q4-v1 (G0), the campaign's proven-coherent promotion incumbent, not
the BF16 parent -- the parent has no native execution path here, and comparing against the
incumbent is the decision actually being made when an artifact is promoted.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile

BIN = "workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy"
TOK = "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json"
RUNS = "workspace/campaign/records/runs/qwen38-27b"

PROMPTS = [
 "The capital of France is", "In Python, a list comprehension looks like",
 "The derivative of x squared with respect to x is", "To reverse a string in Python you can",
 "The largest planet in the solar system is", "A JSON object begins with the character",
 "The time complexity of binary search is", "Water boils at a temperature of",
 "To create a new git branch you run", "The chemical formula for table salt is",
 "A function that calls itself is called", "The first ten prime numbers are",
 "In SQL, to select every column you write", "The capital city of Japan is",
 "To install a Python package you run", "An HTTP status code of 404 means",
 "The square root of 144 is", "In Rust, ownership means that",
 "A binary tree node typically contains", "The speed of light is approximately",
 "To count lines in a file on Linux you use", "The atomic number of carbon is",
 "A hash table provides average lookup time of", "The author of the Iliad is",
 "To exit vim you type", "In statistics, the mean of 2, 4 and 6 is",
 "The main function in C returns type", "A palindrome is a word that",
 "The currency of Japan is called", "To make an HTTP request in Python you can use",
 "The number of bits in a byte is", "A deadlock occurs when",
]


def gen(root, prompts, max_new, max_seq):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(prompts) + "\n")
        pf = fh.name
    r = subprocess.run([BIN, "--artifact-root", os.path.join(RUNS, root), "--tokenizer", TOK,
                        "--prompts-file", pf, "--max-new-tokens", str(max_new),
                        "--max-seq-len", str(max_seq), "--raw-prompt",
                        "--out", tempfile.mktemp(suffix=".json")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{root}: exit {r.returncode}\n{r.stderr[-600:]}")
    toks, fb = [], 0
    for line in r.stdout.splitlines():
        if line.startswith("generated_token_ids="):
            toks.append(json.loads(line.split("=", 1)[1]))
        elif line.startswith("FALLBACKS:"):
            fb += int(line.split(":")[1])
    return toks, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="uniform-q4-v1")
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    ref, fb = gen(a.reference, PROMPTS, a.max_new_tokens, a.max_seq_len)
    assert fb == 0, f"reference took {fb} fallbacks; the comparison would not be native"
    n_tok = sum(len(t) for t in ref)
    print(f"reference {a.reference}: {len(ref)} prompts, {n_tok} generated tokens, FALLBACKS 0")

    # self-consistency: the reference against itself must be identical, or greedy is not
    # deterministic here and every number below is noise
    ref2, _ = gen(a.reference, PROMPTS, a.max_new_tokens, a.max_seq_len)
    same = sum(1 for x, y in zip(ref, ref2) if x == y)
    print(f"determinism control: reference vs itself {same}/{len(ref)} sequences identical"
          f"  {'OK' if same == len(ref) else 'FAILED -- greedy is not deterministic, stop'}")
    if same != len(ref):
        return 1

    out = {}
    print(f"\n{'candidate':<26}{'tok agree':>11}{'seq exact':>11}{'med first div':>14}"
          f"{'never rejoin':>13}{'FALLBACKS':>10}")
    for c in a.candidate:
        cand, cfb = gen(c, PROMPTS, a.max_new_tokens, a.max_seq_len)
        agree = tot = exact = rejoin_never = 0
        firsts = []
        for r_, c_ in zip(ref, cand):
            m = min(len(r_), len(c_))
            eq = [r_[i] == c_[i] for i in range(m)]
            agree += sum(eq); tot += m
            if r_ == c_:
                exact += 1
                continue
            fd = eq.index(False) if False in eq else m
            firsts.append(fd)
            if not any(eq[fd + 1:]):
                rejoin_never += 1
        firsts.sort()
        med = firsts[len(firsts) // 2] if firsts else None
        out[c] = {"token_agreement": agree / max(tot, 1), "sequences_exact": exact,
                  "sequences": len(ref), "median_first_divergence": med,
                  "never_rejoin": rejoin_never, "fallbacks": cfb}
        print(f"{c:<26}{agree/max(tot,1):>11.4f}{f'{exact}/{len(ref)}':>11}"
              f"{(med if med is not None else '-'):>14}{rejoin_never:>13}{cfb:>10}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
