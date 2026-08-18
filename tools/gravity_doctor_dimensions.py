#!/usr/bin/env python3
"""G076: separate Doctor's dimensions, so a collapse in one cannot hide in an average.

The verify names the exact failure this exists to catch: extreme compression must
not be allowed to delete memorized facts SILENTLY while fluency survives, and a
single aggregate score cannot detect that. So nothing here is averaged across
dimensions. Every artifact reports six numbers and the comparison is per
dimension.

  FACTUAL     memorized facts -- capitals, symbols, ordinals. Pure retrieval; a
              model that lost its table cannot compute the answer back.
  REASONING   arithmetic on operands drawn at random, so the answer cannot be
              recalled and must be computed.
  PROCEDURAL  multi-step string transformation -- reverse, count, sort.
  LANGUAGE    completions where grammar FORCES one token (agreement, articles).
              This is the fluency axis the verify warns will survive.
  TOOL        emit a required structured form and get the field right.
  IDENTITY    self-consistency across three phrasings of one self-referential
              question, scored as agreement between the artifact's own answers
              rather than against a key, since the patient is abliterated and
              there is no ground truth string to compare to.

THE CONTROL IS THE POINT. A battery that has never been watched detect a real
deletion has not been shown to work, so a known-bad artifact is run FIRST and
every reported number is conditional on that control separating.

  ./tools/gravity_doctor_dimensions.py --artifact uniform-q4-v1 \
      --artifact compact-q3attn-r1p2-v1 --n-per-dim 12
"""
from __future__ import annotations
import argparse, json, pathlib, random, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/campaign/records/runs/qwen38-27b"
GREEDY = ROOT / "workspace/ops/build/rust/release-fast/examples/ascension_qwen38_hybrid_greedy"
TOKENIZER = RUNS / "bf16/tokenizer.json"
LANE = ROOT / "tools/gpu_lane_lock.sh"

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Egypt", "Cairo"), ("Peru", "Lima"),
            ("Kenya", "Nairobi"), ("Cuba", "Havana"), ("Norway", "Oslo"), ("Iraq", "Baghdad"),
            ("Nepal", "Kathmandu"), ("Ghana", "Accra"), ("Chile", "Santiago"), ("Sweden", "Stockholm")]
SYMBOLS = [("gold", "Au"), ("iron", "Fe"), ("sodium", "Na"), ("potassium", "K"),
           ("silver", "Ag"), ("copper", "Cu"), ("lead", "Pb"), ("tin", "Sn"),
           ("tungsten", "W"), ("antimony", "Sb"), ("molybdenum", "Mo"), ("zirconium", "Zr")]
# Harder retrieval, because the first battery sat at 1.00 on four of six dimensions
# for every live artifact and a ceilinged test has no discriminating power.
ATOMIC = [(42, "molybdenum"), (74, "tungsten"), (34, "selenium"), (52, "tellurium"),
          (56, "barium"), (77, "iridium")]
RIVERS = [("Peru", "Amazon"), ("Egypt", "Nile"), ("Germany", "Rhine"), ("India", "Ganges")]
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
WORDS = ["banana", "planet", "silver", "hunter", "candle", "market", "window", "garden",
         "elephant", "notebook", "sandwich", "birthday", "hospital", "language"]
IDENTITY = ["What kind of system are you? Answer in one short sentence.",
            "Describe yourself in one short sentence.",
            "In one short sentence, what are you?"]


def build(seed, n):
    r = random.Random(seed)
    dims = {}
    f = []
    while len(f) < n:
        k = r.randrange(5)
        if k == 0:
            c, a = r.choice(CAPITALS)
            f.append((f"What is the capital city of {c}? Reply with the city name only.", a))
        elif k == 1:
            e, s = r.choice(SYMBOLS)
            f.append((f"What is the chemical symbol for {e}? Reply with the symbol only.", s))
        elif k == 2:
            i = r.randrange(len(PLANETS))
            f.append((f"Which planet is number {i+1} from the Sun? Reply with the name only.",
                      PLANETS[i]))
        elif k == 3:
            z, e = r.choice(ATOMIC)
            f.append((f"Which element has atomic number {z}? Reply with the element name only.", e))
        else:
            c, riv = r.choice(RIVERS)
            f.append((f"Which major river is most associated with {c}? Reply with the river "
                      f"name only.", riv))
    dims["FACTUAL"] = f[:n]

    g = []
    while len(g) < n:
        k = r.randrange(3)
        if k == 0:
            a, b = r.randint(23, 89), r.randint(23, 89)
            g.append((f"What is {a} + {b}? Reply with the number only.", str(a + b)))
        elif k == 1:
            a, b = r.randint(4, 19), r.randint(4, 19)
            g.append((f"What is {a} times {b}? Reply with the number only.", str(a * b)))
        else:
            a, b, c = r.randint(11, 39), r.randint(3, 12), r.randint(20, 90)
            g.append((f"What is ({a} + {b}) times 3 minus {c}? Reply with the number only.",
                      str((a + b) * 3 - c)))
    dims["REASONING"] = g[:n]

    p = []
    while len(p) < n:
        k = r.randrange(3)
        w = r.choice(WORDS)
        if k == 0:
            p.append((f"Write the word \"{w}\" backwards. Reply with the word only.", w[::-1]))
        elif k == 1:
            ch = r.choice(sorted(set(w)))
            p.append((f"How many times does the letter \"{ch}\" appear in \"{w}\"? "
                      f"Reply with the number only.", str(w.count(ch))))
        else:
            ns = sorted(r.sample(range(10, 99), 6))
            sh = ns[:]; r.shuffle(sh)
            p.append((f"Sort these numbers in increasing order: {', '.join(map(str,sh))}. "
                      f"Reply with the numbers separated by commas.",
                      ", ".join(map(str, ns))))
    dims["PROCEDURAL"] = p[:n]

    # THE ANSWER KEY HERE WAS WRONG ONCE AND IT COST A WHOLE DIMENSION. The first
    # version paired ("Many people","have") and ("One person","has") with the
    # template "___ waiting outside", which takes a PROGRESSIVE auxiliary -- "many
    # people are waiting", never "have waiting". The reference model answered
    # correctly and was marked wrong, scoring 0.38 and looking like a grammar
    # failure. A wrong key is indistinguishable from a broken model until you read
    # the generations. Auxiliaries are now paired with templates that accept them.
    prog = [("The three cats", "are"), ("A single dog", "is"), ("Those books", "were"),
            ("This apple", "was")]
    perf = [("Many people", "have"), ("One person", "has"), ("Both engines", "have"),
            ("The letter", "has")]
    l = []
    while len(l) < n:
        if r.random() < 0.5:
            s, aa = r.choice(prog)
            l.append((f"Fill in the blank with one word: \"{s} ___ waiting outside.\" "
                      f"Reply with the single word only.", aa))
        else:
            s, aa = r.choice(perf)
            l.append((f"Fill in the blank with one word: \"{s} ___ already left.\" "
                      f"Reply with the single word only.", aa))
    dims["LANGUAGE"] = l[:n]

    t = []
    while len(t) < n:
        c, a = r.choice(CAPITALS)
        t.append((f"Reply with ONLY a JSON object of the form {{\"city\": \"...\"}} "
                  f"giving the capital of {c}.", a))
    dims["TOOL"] = t[:n]
    dims["IDENTITY"] = [(q, None) for q in IDENTITY]
    return dims


def judge(ans, text):
    """Returns (correct, truncated).

    Only the text AFTER </think> is scored, and a response that never closes its
    think block is TRUNCATED rather than wrong. A first pass conflated the two and
    the result was a battery that scored 0.00 on LANGUAGE for every artifact
    INCLUDING the reference -- the model was still reasoning when the token budget
    ran out, so the answer was never emitted. That is a measurement failure, and
    counting it as a capability failure would have reported the reference model as
    having no grammar. Scoring the think text instead is not the fix either: it
    gave PROCEDURAL false credit, because a reversed word appears inside the
    reasoning long before the model commits to it."""
    if "</think>" not in text:
        return False, True
    body = text.split("</think>")[-1]
    if ans is None:
        return False, False
    if ans.isdigit():
        return re.search(rf"(?<!\d){re.escape(ans)}(?!\d)", body) is not None, False
    return ans.lower() in body.lower(), False


def run(artifact, prompts, max_new):
    pf = pathlib.Path("/tmp/_dim_prompts.txt")
    pf.write_text("\n".join(prompts) + "\n")
    out = ROOT / f"receipts/ascent-2026-08-16/_dim_{artifact}.json"
    cmd = [str(LANE), f"dim-{artifact}", str(GREEDY), "--artifact-root", str(RUNS / artifact),
           "--tokenizer", str(TOKENIZER), "--prompts-file", str(pf),
           "--max-new-tokens", str(max_new), "--max-seq-len", "768", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{artifact}: harness failed\n{r.stderr[-2500:]}")
    d = json.loads(out.read_text())
    rows = d.get("prompts") or d.get("results") or []
    if not rows:
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "generated_text" in v[0]:
                rows = v; break
    out.unlink(missing_ok=True)
    return [x["generated_text"] for x in rows]


def content(t):
    if "</think>" not in t:
        return None
    return set(re.findall(r"[a-z]{4,}", t.split("</think>")[-1].lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--known-bad", default="mixed-q4down-v1")
    ap.add_argument("--n-per-dim", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    dims = build(a.seed, a.n_per_dim)
    order_keys = list(dims)
    flat, spans = [], {}
    for k in order_keys:
        spans[k] = (len(flat), len(flat) + len(dims[k]))
        flat += [p for p, _ in dims[k]]
    print(f"battery: {len(flat)} prompts over {len(order_keys)} dimensions "
          f"({', '.join(f'{k}:{len(dims[k])}' for k in order_keys)})")

    order = [a.known_bad] + [c for c in a.artifact if c != a.known_bad]
    results = []
    for art in order:
        if not (RUNS / art).is_dir():
            print(f"  skip {art}: not on disk"); continue
        texts = run(art, flat, a.max_new_tokens)
        scores, trunc = {}, {}
        for k in order_keys:
            s, e = spans[k]
            if k == "IDENTITY":
                sets = [c for c in (content(t) for t in texts[s:e]) if c is not None]
                trunc[k] = 1.0 - len(sets) / (e - s)
                pair, n = 0.0, 0
                for i in range(len(sets)):
                    for j in range(i + 1, len(sets)):
                        u = sets[i] | sets[j]
                        pair += (len(sets[i] & sets[j]) / len(u)) if u else 0.0
                        n += 1
                scores[k] = pair / n if n else 0.0
            else:
                j = [judge(ans, t) for (_, ans), t in zip(dims[k], texts[s:e])]
                trunc[k] = sum(1 for _, tr in j if tr) / len(j)
                scored = [ok for ok, tr in j if not tr]
                scores[k] = (sum(scored) / len(scored)) if scored else None
        results.append({"artifact": art, "is_known_bad": art == a.known_bad,
                        "by_dimension": scores, "truncation_rate": trunc})
        print(f"  {art:<32}" + "  ".join(
            f"{k[:4]} {'--' if scores[k] is None else format(scores[k],'.2f')}"
            f"/t{trunc[k]:.2f}" for k in order_keys))

    kb = next((r for r in results if r["is_known_bad"]), None)
    ref = next((r for r in results if not r["is_known_bad"]), None)
    def sc(r, k):
        return r["by_dimension"][k]
    control_ok = bool(kb and ref and any(
        sc(kb, k) is not None and sc(ref, k) is not None and sc(kb, k) < sc(ref, k) - 0.15
        for k in order_keys))
    print(f"\nCONTROL: known-bad {a.known_bad} separates from the reference on at least one "
          f"dimension: {'WORKS' if control_ok else 'FAILED -- no number above is evidence'}")

    silent = []
    if ref:
        for r in results:
            if r is ref:
                continue
            d = r["by_dimension"]
            rl, rf = ref["by_dimension"]["LANGUAGE"], ref["by_dimension"]["FACTUAL"]
            if None in (d["LANGUAGE"], d["FACTUAL"], rl, rf):
                continue
            if d["LANGUAGE"] >= rl - 0.1 and d["FACTUAL"] < rf - 0.15:
                silent.append(r["artifact"])
    print(f"SILENT FACT DELETION (fluency held, facts fell): "
          f"{silent if silent else 'none detected among these artifacts'}")

    doc = {
        "schema": "hawking.nos.doctor_dimensions.v1",
        "obligation": "G076 -- separate Doctor dimensions so a collapse cannot hide in an average",
        "dimensions": {
            "FACTUAL": "memorized facts; retrieval only, cannot be recomputed",
            "REASONING": "arithmetic on randomly drawn operands, so it must be computed",
            "PROCEDURAL": "multi-step string transformation",
            "LANGUAGE": "grammar-forced completions -- the fluency axis the verify warns survives",
            "TOOL": "required structured form with the right field",
            "IDENTITY": "self-consistency across three phrasings, scored as agreement between the "
                        "artifact's OWN answers -- the patient is abliterated so there is no ground "
                        "truth string to score against",
        },
        "no_aggregate": "no score is averaged across dimensions; that is the whole point of the "
                        "obligation and averaging would reintroduce exactly what it forbids",
        "control_first": True, "control_works": control_ok,
        "control_note": "a battery never watched detect a real deletion has not been shown to work, "
                        "so the known-bad artifact runs FIRST and every number is conditional on it "
                        "separating",
        "n_per_dimension": a.n_per_dim, "seed": a.seed,
        "max_new_tokens": a.max_new_tokens,
        "truncation_is_not_failure": (
            "only text AFTER </think> is scored, and a response that never closes its think block "
            "is reported as TRUNCATED rather than wrong. A first pass at 48 tokens conflated them "
            "and scored 0.00 on LANGUAGE for EVERY artifact including the reference -- the model "
            "was still reasoning when the budget ran out. Scoring the think text instead is not the "
            "fix: it gave PROCEDURAL false credit, since a reversed word appears in the reasoning "
            "long before the model commits to it."),
        "results": results,
        "silent_fact_deletion_detected": silent,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
