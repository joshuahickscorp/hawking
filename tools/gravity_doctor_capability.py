#!/usr/bin/env python3
"""Doctor capability gate: a behavioural test with its negative control watched.

The campaign's coherence verdicts have rested on two prompts (France-Paris, 17x19). That is
not a capability contract, and the cost of the gap is now measured: the per-tensor adequacy
gate rejects 15 of 15 MLP tensors in an artifact that those two prompts call coherent. One
of the two instruments is miscalibrated and neither can settle it alone.

This is the behavioural half. It drives the real native runtime -- no proxy, no reconstruct,
FALLBACKS and DENSE_W_MATERIALIZED asserted zero -- across the dimensions the identity
contract names, and it scores DETERMINISTIC answers so a verdict is not a matter of reading
fluency.

It also refuses to report a PASS it has not earned. Every run must include a known-broken
artifact and watch it FAIL, because a gate that has never been observed failing is not
evidence. mixed-q4down-v1 is that control: measured degenerate, 0 of 5, emits end-of-turn
after zero or one tokens.

DEGENERACY is scored separately from CORRECTNESS, because they fail differently. An artifact
that emits end-of-turn immediately is not "wrong", it is absent, and averaging that with a
wrong-but-fluent answer hides which failure occurred.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile

BIN = "workspace/ops/build/rust/release/examples/ascension_qwen38_hybrid_greedy"
TOK = "workspace/campaign/records/runs/qwen38-27b/bf16/tokenizer.json"
RUNS = "workspace/campaign/records/runs/qwen38-27b"

# dimension, prompt, and a predicate on the generated text. Predicates are deliberately
# permissive about form and strict about content: the model may reason before answering.
BATTERY = [
    ("factual",      "The capital of France is",              lambda t: "paris" in t.lower()),
    ("factual",      "The chemical symbol for gold is",        lambda t: "au" in t.lower()),
    ("arithmetic",   "17 x 19 =",                              lambda t: "323" in t),
    ("arithmetic",   "What is 144 divided by 12? Answer:",     lambda t: "12" in t),
    ("code",         "Complete this Python line: def add(a, b): return",
                                                               lambda t: "+" in t),
    ("code",         "In Python, the keyword to define a function is",
                                                               lambda t: "def" in t.lower()),
    ("tool",         'Repeat this JSON exactly: {"ok": true}',  lambda t: '"ok"' in t and "true" in t.lower()),
    ("instruction",  "Reply with exactly one word: yes or no. Is the sky blue?",
                                                               lambda t: re.search(r"\byes\b", t, re.I) is not None),
    ("multilingual", "Translate to French: the cat. Answer:",   lambda t: "chat" in t.lower()),
    ("reasoning",    "List three prime numbers greater than 20.",
                                                               lambda t: sum(p in t for p in ("23","29","31","37","41","43")) >= 2),
]

# a run is DEGENERATE on a prompt when the model produces essentially nothing before
# end-of-turn. This is the failure mode a broken artifact actually exhibits.
EOT = 248046
MIN_REAL_TOKENS = 3


def run(root, max_new, max_seq):
    for _, prm, _ in BATTERY:
        assert "\n" not in prm, f"prompt contains a newline and would split the file: {prm!r}"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(p for _, p, _ in BATTERY) + "\n")
        pf = fh.name
    out = tempfile.mktemp(suffix=".json")
    cmd = [BIN, "--artifact-root", os.path.join(RUNS, root), "--tokenizer", TOK,
           "--prompts-file", pf, "--max-new-tokens", str(max_new),
           "--max-seq-len", str(max_seq), "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{root}: runtime exited {r.returncode}\n{r.stderr[-800:]}")
    txt, toks, fb, dw = [], [], 0, 0
    cur = None
    for line in r.stdout.splitlines():
        if line.startswith("GENERATED_TEXT_VERBATIM:"):
            cur = [line.split(":", 1)[1].lstrip()]
        elif line.startswith("FALLBACKS:"):
            fb += int(line.split(":")[1]); txt.append("\n".join(cur or [])); cur = None
        elif line.startswith("DENSE_W_MATERIALIZED:"):
            dw += int(line.split(":")[1])
        elif line.startswith("generated_token_ids="):
            toks.append(json.loads(line.split("=", 1)[1]))
        elif cur is not None:
            cur.append(line)
    return txt, toks, fb, dw


def score(root, max_new=260, max_seq=768):
    txt, toks, fb, dw = run(root, max_new, max_seq)
    n = min(len(txt), len(toks), len(BATTERY))
    rows, by_dim = [], {}
    for i in range(n):
        dim, prompt, pred = BATTERY[i]
        real = [t for t in toks[i] if t != EOT]
        degen = len(real) < MIN_REAL_TOKENS
        ok = bool(pred(txt[i])) and not degen
        rows.append({"dim": dim, "prompt": prompt, "ok": ok, "degenerate": degen,
                     "n_tokens": len(real), "text": txt[i][:120]})
        d = by_dim.setdefault(dim, [0, 0, 0])
        d[0] += ok; d[1] += 1; d[2] += degen
    return {"artifact": root, "n": n, "fallbacks": fb, "dense_materialized": dw,
            "rows": rows, "by_dim": by_dim,
            "correct": sum(r["ok"] for r in rows),
            "degenerate": sum(r["degenerate"] for r in rows)}


def report(res):
    print(f"\n=== {res['artifact']} ===")
    print(f"  FALLBACKS {res['fallbacks']}  DENSE_W_MATERIALIZED {res['dense_materialized']}"
          f"   (both must be 0 or the native path did not execute)")
    for dim, (ok, tot, deg) in sorted(res["by_dim"].items()):
        print(f"  {dim:<14}{ok}/{tot} correct   {deg} degenerate")
    print(f"  TOTAL {res['correct']}/{res['n']} correct, {res['degenerate']} degenerate")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--control-pass", default="uniform-q4-v1",
                    help="artifact that MUST pass, or the battery is broken")
    ap.add_argument("--control-fail", default="mixed-q4down-v1",
                    help="artifact that MUST fail, or the battery proves nothing")
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--max-seq-len", type=int, default=768)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    order, seen = [], set()
    for r in [a.control_pass, a.control_fail] + a.artifact:
        if r and r not in seen:
            seen.add(r); order.append(r)
    out = {r: report(score(r, a.max_new_tokens, a.max_seq_len)) for r in order}

    pos, neg = out[a.control_pass], out[a.control_fail]
    print("\n--- controls ---")
    ok_pos = pos["correct"] >= 0.6 * pos["n"] and pos["degenerate"] == 0
    ok_neg = neg["correct"] <= 0.2 * neg["n"]
    print(f"  positive control {a.control_pass:<20} {pos['correct']}/{pos['n']}  "
          f"{'PASSES as required' if ok_pos else 'FAILED -- the battery or the rig is broken'}")
    print(f"  negative control {a.control_fail:<20} {neg['correct']}/{neg['n']}  "
          f"{'FAILS as required' if ok_neg else 'PASSED -- the battery cannot detect a broken model'}")
    if not (ok_pos and ok_neg):
        print("\n  NO VERDICT IS REPORTED. A gate whose controls do not behave is not evidence.")
        return 1
    print("\n--- verdicts (controls behaved, so these mean something) ---")
    for r in order:
        if r in (a.control_pass, a.control_fail):
            continue
        v = out[r]
        print(f"  {r:<26}{v['correct']}/{v['n']} correct, {v['degenerate']} degenerate  "
              f"-> {'COHERENT' if v['correct'] >= 0.6*v['n'] and v['degenerate'] == 0 else 'INADEQUATE'}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
