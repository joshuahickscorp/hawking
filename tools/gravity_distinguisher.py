#!/usr/bin/env python3
"""G048: an adversarial distinguisher, scored at a stated query budget.

A fixed benchmark can only score the directions its prompts happen to excite, and
G046/G047 measured why that is fatal: the observed activation span keeps growing
with every prompt family added and never saturates, and the operator's gain on
directions the capture NEVER visits is 0.71-1.03x its gain on observed ones. A
ten-item battery is not a small version of the right test, it is a different test.

So the objective here is not "score the candidate" but "find a prompt where the
candidate and the teacher disagree on protected capability". Prompts are GENERATED
from seeded templates rather than listed, so the budget k buys draws from a space
instead of re-running the same items.

A distinguishing event requires BOTH halves: the teacher answers correctly AND the
candidate does not. A prompt both get wrong says nothing about equivalence, and a
prompt where they merely produce different text says nothing either -- G004
measured greedy divergence ranking candidates OPPOSITELY to capability, so token
disagreement is not the signal.

The verify line is the discipline: run against a KNOWN-BAD candidate first and
watch it succeed. A distinguisher that has never found a real divergence has not
been shown to work, so the known-bad run is not a formality, it is the control
that makes the other numbers mean anything.

  ./tools/gravity_distinguisher.py --candidate mixed-q4down-v1 \
      --candidate compact-q3attn-r1p2-v1 --budget 32
"""
from __future__ import annotations
import argparse, json, pathlib, random, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
GREEDY = ROOT / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
TOKENIZER = RUNS / "bf16/tokenizer.json"
LANE = ROOT / "tools/gpu_lane_lock.sh"
TEACHER = "uniform-q4-v1"

WORDS = ["banana", "elephant", "keyboard", "mountain", "triangle", "umbrella",
         "notebook", "sandwich", "computer", "birthday", "hospital", "language"]


def generate(seed: int, n: int):
    """Seeded draws from a generated space, not a fixed list."""
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        kind = rng.choice(["add", "mul", "sub", "len", "rev", "count"])
        if kind == "add":
            a, b = rng.randint(11, 89), rng.randint(11, 89)
            items.append((f"What is {a} + {b}? Reply with the number only.", str(a + b)))
        elif kind == "mul":
            a, b = rng.randint(3, 19), rng.randint(3, 19)
            items.append((f"What is {a} times {b}? Reply with the number only.", str(a * b)))
        elif kind == "sub":
            a, b = rng.randint(50, 199), rng.randint(11, 49)
            items.append((f"What is {a} - {b}? Reply with the number only.", str(a - b)))
        elif kind == "len":
            w = rng.choice(WORDS)
            items.append((f"How many letters are in the word \"{w}\"? Reply with the number only.",
                          str(len(w))))
        elif kind == "rev":
            w = rng.choice(WORDS)[:5]
            items.append((f"Write the word \"{w}\" backwards. Reply with the word only.",
                          w[::-1]))
        else:
            w = rng.choice(WORDS)
            ch = rng.choice(sorted(set(w)))
            items.append((f"How many times does the letter \"{ch}\" appear in \"{w}\"? "
                          f"Reply with the number only.", str(w.count(ch))))
    return items[:n]


def correct(answer: str, text: str) -> bool:
    """Deterministic check. Numbers must appear as a standalone token so that
    '7' does not match inside '17'."""
    body = text.split("</think>")[-1] if "</think>" in text else text
    if answer.isdigit():
        return re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", body) is not None
    return answer.lower() in body.lower()


def run(artifact: str, items, max_new: int) -> list[dict]:
    pf = ROOT / f"/tmp/_dist_prompts.txt"
    pf.write_text("\n".join(p for p, _ in items) + "\n")
    out = ROOT / f"receipts/ascent-2026-08-16/_dist_{artifact}.json"
    cmd = [str(LANE), f"dist-{artifact}", str(GREEDY),
           "--artifact-root", str(RUNS / artifact), "--tokenizer", str(TOKENIZER),
           "--prompts-file", str(pf), "--max-new-tokens", str(max_new),
           "--max-seq-len", "512", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{artifact}: harness failed\n{r.stderr[-2000:]}")
    d = json.loads(out.read_text())
    rows = d.get("prompts") or d.get("results") or []
    if not rows:
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "generated_text" in v[0]:
                rows = v
                break
    out.unlink(missing_ok=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="append", required=True)
    ap.add_argument("--known-bad", default="mixed-q4down-v1")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    items = generate(a.seed, a.budget)
    print(f"query budget k={a.budget}, generated from seed {a.seed}")

    tr = run(TEACHER, items, a.max_new_tokens)
    t_ok = [correct(ans, tr[i]["generated_text"]) for i, (_, ans) in enumerate(items)]
    print(f"teacher {TEACHER}: correct on {sum(t_ok)}/{len(items)}")

    order = [a.known_bad] + [c for c in a.candidate if c != a.known_bad]
    results = []
    for cand in order:
        cr = run(cand, items, a.max_new_tokens)
        events, first = [], None
        for i, (prompt, ans) in enumerate(items):
            c_ok = correct(ans, cr[i]["generated_text"])
            if t_ok[i] and not c_ok:
                events.append({"query": i, "prompt": prompt, "expected": ans,
                               "candidate_said": cr[i]["generated_text"][:160]})
                if first is None:
                    first = i + 1
        score = len(events) / len(items)
        results.append({"candidate": cand, "is_known_bad": cand == a.known_bad,
                        "distinguishing_events": len(events), "budget": len(items),
                        "distinguishability": score, "queries_to_first_event": first,
                        "teacher_correct": sum(t_ok), "events": events[:6]})
        print(f"  {cand:<30} distinguishability {score:.3f} "
              f"({len(events)}/{len(items)}), first at query "
              f"{first if first else '-- none found'}")

    kb = next(r for r in results if r["is_known_bad"])
    control_ok = kb["distinguishing_events"] > 0
    print(f"\nCONTROL: known-bad {kb['candidate']} distinguished at query "
          f"{kb['queries_to_first_event']} -> "
          f"{'WORKS' if control_ok else 'FAILED -- no result below is evidence'}")

    doc = {
        "schema": "hawking.nos.h_equivalence_distinguisher.v1",
        "obligation": "G048 -- adversarial distinguisher instead of a fixed benchmark",
        "teacher": TEACHER,
        "query_budget": a.budget, "seed": a.seed, "max_new_tokens": a.max_new_tokens,
        "prompt_space": "seeded generation over six deterministic templates (add, mul, sub, "
                        "letter-count, reverse, character-count) with randomised operands and "
                        "words -- the budget buys DRAWS FROM A SPACE, not reruns of a list",
        "distinguishing_event": "teacher correct AND candidate wrong. A prompt both get wrong says "
                                "nothing about equivalence, and mere token disagreement says "
                                "nothing either -- G004 measured greedy divergence ranking "
                                "candidates OPPOSITELY to capability",
        "control_ran_first": True,
        "control_works": control_ok,
        "results": results,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
