#!/usr/bin/env python3.12
"""Assemble a CURATED MULTI-DOMAIN calibration corpus for the condense doctor.

WHY DIVERSITY MATTERS (activation coverage):
  AWQ derives per-input-channel importance from mean|x| over the calib pass
  (awq_bake.py / audit_ladder.capture_sigma), and KD/LoRA train against the
  teacher's logits over the SAME text. Both are only as good as the activation
  distribution the calib corpus elicits. A single-domain corpus (e.g. the current
  scratch/calib_corpus.txt = wikitext prose) lights up the channels that prose
  uses and leaves code/math/dialogue/structured channels DARK — those channels
  then get scaled/quantized as if unimportant, and the residual the doctor must
  heal is dominated by the domains the calib never saw. A curated mix that spans
  code + prose + math + dialogue + structured/JSON exercises a far wider set of
  input channels per linear, so:
    * AWQ's sigma^alpha protects channels that matter across the model's real use,
      not just one register; and
    * KD sees teacher behavior on the full input manifold, shrinking the gap the
      doctor has to close on held-out (multi-domain) eval.
  Net: same bpw, lower output-space degradation, because the recovery signal is
  representative instead of prose-skewed.

SOURCES (all local — no download):
  * code      : crates/**/*.rs  (the repo's own Rust)
  * prose     : docs/**/*.md + existing scratch/calib_corpus*.txt (wikitext-ish)
  * math      : synthetic arithmetic / algebra / number-theory statements
  * dialogue  : synthetic multi-turn chat-template-shaped exchanges
  * structured: synthetic JSON / config / table rows
  The synthetic generators are DETERMINISTIC (fixed seed) and clearly documented
  so the corpus is reproducible. If a local source is thin, its share is topped up
  from the synthetic mix and a note is printed to stderr.

Usage:
  python3.12 tools/condense/calib_build.py [out.txt] [target_bytes] [seed]
  # defaults: scratch/calib_multidomain.txt  ~2_000_000 bytes  seed=0

Then point the bakers at it:
  DOCTOR_CALIB=scratch/calib_multidomain.txt PPL_TEXT=... <baker>

Prints one JSON line with the realized per-domain byte breakdown.
"""
import sys, os, re, json, glob, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "scratch", "calib_multidomain.txt")
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0

# Domain budget (fractions of TARGET). Roughly even with a prose lead because
# prose is the cheapest to source cleanly and anchors the language model.
WEIGHTS = {
    "prose": 0.28,
    "code": 0.24,
    "math": 0.16,
    "dialogue": 0.16,
    "structured": 0.16,
}

rng = random.Random(SEED)


def log(m):
    print(m, file=sys.stderr)
    sys.stderr.flush()


def _read_files(patterns, limit_bytes, per_file_cap=40_000):
    """Concatenate matching files (shuffled, deterministic) up to limit_bytes."""
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(ROOT, pat), recursive=True))
    paths = sorted(set(paths))
    rng.shuffle(paths)
    chunks, total = [], 0
    for p in paths:
        if total >= limit_bytes:
            break
        try:
            t = open(p, errors="ignore").read()
        except OSError:
            continue
        t = t.strip()
        if len(t) < 80:
            continue
        t = t[:per_file_cap]
        chunks.append(t)
        total += len(t)
    return "\n\n".join(chunks), total


# ---------------- real local sources ----------------
def collect_code(limit):
    txt, got = _read_files(["crates/**/*.rs"], limit)
    return txt, got


def collect_prose(limit):
    # repo docs first, then top up from the existing wikitext-ish calib corpora
    txt, got = _read_files(["docs/**/*.md", "docs/*.md"], limit)
    if got < limit:
        for fallback in ("scratch/calib_corpus_big.txt", "scratch/calib_corpus.txt"):
            fp = os.path.join(ROOT, fallback)
            if os.path.exists(fp):
                extra = open(fp, errors="ignore").read()[: (limit - got)]
                txt += "\n\n" + extra
                got += len(extra)
            if got >= limit:
                break
    return txt, got


# ---------------- deterministic synthetic generators ----------------
def gen_math(limit):
    """Arithmetic, algebra, and number-theory statements in natural+symbolic form."""
    out, ops = [], [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
                     ("*", lambda a, b: a * b)]
    while sum(len(x) for x in out) < limit:
        kind = rng.randint(0, 3)
        if kind == 0:
            a, b = rng.randint(2, 999), rng.randint(2, 999)
            sym, fn = rng.choice(ops)
            out.append(f"Compute {a} {sym} {b}. The result is {fn(a, b)}.")
        elif kind == 1:
            a, b, x = rng.randint(2, 30), rng.randint(1, 50), rng.randint(1, 20)
            out.append(f"Solve {a}x + {b} = {a * x + b} for x. Then x = {x}, "
                       f"since {a}*{x} = {a * x} and {a * x} + {b} = {a * x + b}.")
        elif kind == 2:
            n = rng.randint(2, 40)
            fact, k = 1, n
            while k > 1:
                fact *= k
                k -= 1
            out.append(f"The factorial {n}! equals {fact}.")
        else:
            n = rng.randint(2, 200)
            divs = [d for d in range(1, n + 1) if n % d == 0]
            out.append(f"The divisors of {n} are {divs}; it is "
                       f"{'prime' if len(divs) == 2 else 'composite'}.")
    return " ".join(out)[:limit], min(limit, sum(len(x) for x in out))


def gen_dialogue(limit):
    """Multi-turn chat-template-shaped exchanges (ChatML-ish) over varied intents."""
    topics = [
        ("How do I reverse a list in Python?",
         "Use slicing: lst[::-1], or lst.reverse() to mutate in place."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Summarize what a hash map is.",
         "A hash map stores key-value pairs with average O(1) lookup via a hash function."),
        ("Translate 'good morning' to Spanish.", "It is 'buenos dias'."),
        ("Why is the sky blue?",
         "Shorter blue wavelengths scatter more in the atmosphere (Rayleigh scattering)."),
        ("Give me a regex for an email.",
         "A simple one: ^[\\w.+-]+@[\\w-]+\\.[\\w.-]+$ (not RFC-complete)."),
        ("What does quantization do to a model?",
         "It lowers weight precision to shrink memory and bandwidth, trading some accuracy."),
    ]
    out = []
    while sum(len(x) for x in out) < limit:
        q, a = rng.choice(topics)
        sys_msg = rng.choice(["You are a helpful assistant.",
                              "You are a concise expert.",
                              "Answer accurately and briefly."])
        out.append(f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                   f"<|im_start|>user\n{q}<|im_end|>\n"
                   f"<|im_start|>assistant\n{a}<|im_end|>\n")
    return "".join(out)[:limit], min(limit, sum(len(x) for x in out))


def gen_structured(limit):
    """JSON objects, key=value config blocks, and TSV-ish table rows."""
    names = ["alice", "bob", "carol", "dave", "erin", "frank"]
    cities = ["Paris", "Tokyo", "Lima", "Oslo", "Cairo", "Delhi"]
    out = []
    while sum(len(x) for x in out) < limit:
        kind = rng.randint(0, 2)
        if kind == 0:
            obj = {"id": rng.randint(1, 9999), "name": rng.choice(names),
                   "city": rng.choice(cities), "active": rng.choice([True, False]),
                   "score": round(rng.random() * 100, 2),
                   "tags": rng.sample(["a", "b", "c", "d", "e"], rng.randint(1, 3))}
            out.append(json.dumps(obj))
        elif kind == 1:
            out.append(f"[server]\nhost = {rng.choice(cities).lower()}.example.com\n"
                       f"port = {rng.randint(1024, 65535)}\n"
                       f"workers = {rng.randint(1, 32)}\n"
                       f"tls = {str(rng.choice([True, False])).lower()}\n")
        else:
            out.append(f"{rng.choice(names)}\t{rng.randint(18, 80)}\t"
                       f"{rng.choice(cities)}\t{round(rng.random(), 3)}")
    return "\n".join(out)[:limit], min(limit, sum(len(x) for x in out))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    budget = {d: int(TARGET * w) for d, w in WEIGHTS.items()}
    sources = {
        "prose": collect_prose,
        "code": collect_code,
        "math": gen_math,
        "dialogue": gen_dialogue,
        "structured": gen_structured,
    }
    parts, realized = [], {}
    for dom, fn in sources.items():
        txt, got = fn(budget[dom])
        # top up real sources that came up short with synthetic structured filler
        if got < budget[dom] * 0.9 and dom in ("code", "prose"):
            log(f"# {dom}: local source thin ({got}/{budget[dom]} B) -> topping up synthetically")
            filler, _ = gen_structured(budget[dom] - got)
            txt = (txt + "\n\n" + filler) if txt else filler
            got = len(txt)
        header = f"\n\n===== DOMAIN: {dom} =====\n\n"
        parts.append(header + txt)
        realized[dom] = got

    corpus = "".join(parts).strip() + "\n"
    with open(OUT, "w") as f:
        f.write(corpus)

    print(json.dumps({
        "out": OUT, "total_bytes": len(corpus), "target_bytes": TARGET,
        "seed": SEED, "per_domain_bytes": realized,
    }))


if __name__ == "__main__":
    main()
