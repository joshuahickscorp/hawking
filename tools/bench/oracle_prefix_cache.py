#!/usr/bin/env python3
"""Bible Stage-A oracle (§8.1 L1.2) — cross-prompt computation reuse.

Measures the two quantities the Throughput Bible names for the prefix /
semantic cache lever ("THE FIRST MOVE"):

  (a) PREFIX-CACHE hit-rate: for each consecutive request pair, the longest
      common *token* prefix length / request length. A long shared prefix is
      bit-identical reuse — the engine skips that many forward passes. We
      aggregate to a hit-rate distribution (mean/median + buckets) plus the
      energy proxy: fraction of all prompt tokens that fall inside an already-
      cached prefix (= tokens NOT recomputed across the session).

  (b) SEMANTIC-CACHE potential: near-duplicate rate of contexts under an
      embedding-similarity threshold. Embedding is a *local, model-free*
      method — token-set Jaccard and hashed-bigram cosine — so no GPU / model
      load is needed. We report the near-dup rate at several thresholds and,
      because the bible insists on a verify before trusting a near-hit, the
      "verified" rate (near-hit AND exact-prefix-confirmable).

Tokenisation mirrors tools/bench/spec_oracle_on_transcripts.sh: shell out to
$TOKENIZE_BIN (default llama-tokenize) against the Qwen2.5-3B gguf. If that
binary is unavailable we fall back to a whitespace/byte tokenizer so the tool
is never blocked (the fallback is clearly flagged in the output).

TURNKEY on real data. Two input modes:

  # N transcript text files (one served session per file: prompt+completion
  # concatenated, OR a multi-request log — each file is treated as ONE request
  # unless --split-on is given to break a file into a request sequence):
  oracle_prefix_cache.py --texts sess1.txt sess2.txt ...

  # a JSONL of request sequences — each line is one request object; the
  # natural order of lines is the request order of a session:
  oracle_prefix_cache.py --jsonl requests.jsonl
  #   line schema: {"request": "..."}   (also accepts "prompt"/"text"/"content")

No kernel needed to measure. Verdict framing: high prefix overlap on code is
expected; the bible jumps this lever to the front if the hit-rate is high.
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict

DEFAULT_MODEL = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
# GO framing thresholds (fraction of session prompt-tokens served from cache).
GO_FRAC = 0.50      # >=50% of prompt tokens reusable -> front of build queue
MARGINAL_FRAC = 0.25
# Semantic-cache similarity thresholds to sweep.
SEM_THRESHOLDS = (0.95, 0.90, 0.80, 0.70)


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
def find_tokenize_bin():
    """Resolve $TOKENIZE_BIN (default llama-tokenize) on PATH, else None."""
    cand = os.environ.get("TOKENIZE_BIN", "llama-tokenize")
    return cand if shutil.which(cand) else None


_TOK_ID_RE = re.compile(r"\s*(\d+)")


def _parse_id_dump(text):
    ids = []
    for line in text.splitlines():
        m = _TOK_ID_RE.match(line)
        if m:
            ids.append(int(m.group(1)))
    return ids


def tokenize_with_llama(text, model, bin_path):
    """llama-tokenize -m MODEL -f FILE -> list[int]. None on failure."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(text)
        fpath = f.name
    try:
        proc = subprocess.run(
            [bin_path, "-m", model, "-f", fpath],
            capture_output=True, timeout=120)
        if proc.returncode != 0:
            return None
        # decode bytes ourselves with errors="replace": llama-tokenize prints
        # the piece text alongside each id, which may be a partial UTF-8 byte
        # if the input was windowed mid-codepoint. We only read the leading id.
        stdout = proc.stdout.decode("utf-8", errors="replace")
        ids = _parse_id_dump(stdout)
        return ids if ids else None
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        try:
            os.unlink(fpath)
        except OSError:
            pass


_WS_RE = re.compile(r"\S+")


def tokenize_fallback(text):
    """Model-free tokenizer: whitespace words, falling back to bytes if the
    text has almost no whitespace. Deterministic; only used when llama-
    tokenize is unavailable. Flagged in output so numbers aren't mistaken for
    true BPE counts."""
    words = _WS_RE.findall(text)
    # If "words" are huge (e.g. minified), bytes are a better unit.
    if words and (len(text) / max(1, len(words))) > 40:
        return list(text.encode("utf-8", errors="replace"))
    return words


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def hashed_bigram_vec(tokens):
    """Sparse hashed token-bigram multiset (dict hash->count). Model-free
    stand-in for an embedding; cosine over these approximates n-gram overlap
    similarity without any GPU/model."""
    v = defaultdict(float)
    if not tokens:
        return v
    # unigrams + bigrams, so single-token shifts still register similarity.
    for t in tokens:
        v[hash(("1", t)) & 0xFFFFFFFF] += 1.0
    prev = None
    for t in tokens:
        if prev is not None:
            v[hash(("2", prev, t)) & 0xFFFFFFFF] += 1.0
        prev = t
    return v


def cosine(u, v):
    if not u or not v:
        return 0.0
    # iterate smaller dict
    if len(u) > len(v):
        u, v = v, u
    dot = sum(val * v.get(k, 0.0) for k, val in u.items())
    nu = sum(val * val for val in u.values()) ** 0.5
    nv = sum(val * val for val in v.values()) ** 0.5
    return dot / (nu * nv) if nu and nv else 0.0


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def bucketize(fracs):
    buckets = {"0-10%": 0, "10-30%": 0, "30-50%": 0,
               "50-70%": 0, "70-90%": 0, "90-100%": 0}
    for f in fracs:
        pct = f * 100
        if pct < 10:
            buckets["0-10%"] += 1
        elif pct < 30:
            buckets["10-30%"] += 1
        elif pct < 50:
            buckets["30-50%"] += 1
        elif pct < 70:
            buckets["50-70%"] += 1
        elif pct < 90:
            buckets["70-90%"] += 1
        else:
            buckets["90-100%"] += 1
    return buckets


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def load_requests(args):
    """Return (list[str] requests, source_label)."""
    if args.jsonl:
        reqs = []
        for ln in open(args.jsonl, encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                reqs.append(obj)
                continue
            for key in ("request", "prompt", "text", "content"):
                if key in obj and isinstance(obj[key], str):
                    reqs.append(obj[key])
                    break
        return reqs, f"jsonl:{args.jsonl}"
    # --texts
    reqs = []
    for path in args.texts:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if args.split_on:
            parts = [p for p in text.split(args.split_on) if p.strip()]
            reqs.extend(parts)
        else:
            reqs.append(text)
    return reqs, f"texts:{len(args.texts)} file(s)"


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def run(requests, model, tok_bin):
    tokenized = []
    method = "llama-tokenize" if tok_bin else "fallback(ws/byte)"
    used_fallback = tok_bin is None
    for r in requests:
        ids = None
        if tok_bin:
            ids = tokenize_with_llama(r, model, tok_bin)
            if ids is None:
                used_fallback = True
        if ids is None:
            ids = tokenize_fallback(r)
        tokenized.append(ids)
    if used_fallback and tok_bin:
        method = "mixed(llama-tokenize + fallback on failures)"

    n = len(tokenized)
    # (a) prefix metrics on consecutive pairs
    pair_fracs = []
    pair_abs = []
    total_prefix_tokens = 0
    total_prompt_tokens = sum(len(t) for t in tokenized)
    for i in range(1, n):
        prev, cur = tokenized[i - 1], tokenized[i]
        cpl = common_prefix_len(prev, cur)
        frac = cpl / len(cur) if cur else 0.0
        pair_fracs.append(frac)
        pair_abs.append(cpl)
        total_prefix_tokens += cpl

    # Energy proxy: across the whole session, the first request is computed in
    # full; each later request reuses its shared prefix with the immediately
    # preceding one. tokens_recomputed = sum over i>=1 of (len_i - cpl_i)
    # plus len_0. Reuse fraction = cached / total prompt tokens.
    recomputed = (len(tokenized[0]) if n else 0) + sum(
        len(tokenized[i]) - pair_abs[i - 1] for i in range(1, n))
    cached_tokens = total_prompt_tokens - recomputed
    reuse_frac = cached_tokens / total_prompt_tokens if total_prompt_tokens else 0.0

    # (b) semantic near-dup. Compare each request to ALL prior requests (a
    # cache holds history, not just the last one). near-dup if max prior
    # similarity >= threshold. "verified" = near-dup AND some prior shares a
    # non-trivial exact token prefix (>=50% of the shorter), i.e. a real reuse
    # opportunity the verify step would confirm.
    tok_sets = [set(t) for t in tokenized]
    vecs = [hashed_bigram_vec(t) for t in tokenized]
    sem = {f"cos>={th}": 0 for th in SEM_THRESHOLDS}
    sem_jac = {f"jac>={th}": 0 for th in SEM_THRESHOLDS}
    verified = {f"cos>={th}": 0 for th in SEM_THRESHOLDS}
    best_prior_cos = []
    for i in range(1, n):
        max_cos = 0.0
        max_jac = 0.0
        best_j = -1
        for j in range(i):
            c = cosine(vecs[i], vecs[j])
            if c > max_cos:
                max_cos, best_j = c, j
            jac = jaccard(tok_sets[i], tok_sets[j])
            if jac > max_jac:
                max_jac = jac
        best_prior_cos.append(round(max_cos, 4))
        # exact-prefix confirmation against the best semantic match
        confirmable = False
        if best_j >= 0:
            cpl = common_prefix_len(tokenized[best_j], tokenized[i])
            shorter = min(len(tokenized[best_j]), len(tokenized[i])) or 1
            confirmable = (cpl / shorter) >= 0.50
        for th in SEM_THRESHOLDS:
            if max_cos >= th:
                sem[f"cos>={th}"] += 1
                if confirmable:
                    verified[f"cos>={th}"] += 1
            if max_jac >= th:
                sem_jac[f"jac>={th}"] += 1

    pairs = max(0, n - 1)

    def rate(d):
        return {k: round(v / pairs, 3) if pairs else 0.0 for k, v in d.items()}

    result = {
        "oracle": "prefix_cache_semantic_reuse",
        "lever": "§8.1 L1.2 cross-prompt computation reuse",
        "tokenizer": method,
        "used_fallback_tokenizer": used_fallback,
        "n_requests": n,
        "n_consecutive_pairs": pairs,
        "total_prompt_tokens": total_prompt_tokens,
        # (a) PREFIX CACHE
        "prefix_cache": {
            "mean_shared_prefix_frac": round(statistics.fmean(pair_fracs), 4)
            if pair_fracs else 0.0,
            "median_shared_prefix_frac": round(statistics.median(pair_fracs), 4)
            if pair_fracs else 0.0,
            "mean_shared_prefix_tokens": round(statistics.fmean(pair_abs), 1)
            if pair_abs else 0.0,
            "distribution_buckets": bucketize(pair_fracs),
            "session_reuse_frac": round(reuse_frac, 4),
            "cached_tokens": cached_tokens,
            "recomputed_tokens": recomputed,
        },
        # (b) SEMANTIC CACHE
        "semantic_cache": {
            "near_dup_rate_cosine": rate(sem),
            "near_dup_rate_jaccard": rate(sem_jac),
            "verified_near_dup_rate_cosine": rate(verified),
            "mean_best_prior_cosine": round(
                statistics.fmean(best_prior_cos), 4) if best_prior_cos else 0.0,
        },
    }

    # Verdict on the energy proxy (session reuse fraction).
    if reuse_frac >= GO_FRAC:
        verdict = "GO"
    elif reuse_frac >= MARGINAL_FRAC:
        verdict = "MARGINAL"
    else:
        verdict = "NO-GO"
    result["go_thresholds"] = {"GO_frac": GO_FRAC, "MARGINAL_frac": MARGINAL_FRAC}
    result["verdict"] = verdict
    result["note"] = (
        "session_reuse_frac = prompt tokens served from an already-cached "
        "prefix / all prompt tokens (the energy proxy: skipped forward work). "
        "near_dup_rate = consecutive request whose nearest prior context "
        "exceeds the similarity threshold; verified_* additionally require a "
        ">=50% exact token-prefix overlap with that match (the bible's verify "
        "before trusting a semantic hit). GO if reuse_frac>=0.50.")
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--texts", nargs="+", metavar="FILE",
                   help="transcript text files, in session order")
    g.add_argument("--jsonl", metavar="FILE",
                   help="JSONL of request objects, one request per line")
    ap.add_argument("--split-on", default=None,
                    help="with --texts: split each file into a request "
                         "sequence on this delimiter (e.g. a turn marker)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="gguf for llama-tokenize (default %(default)s)")
    ap.add_argument("--out", default="reports/oracle/prefix_cache.json")
    args = ap.parse_args()

    requests, src = load_requests(args)
    if len(requests) < 2:
        sys.exit(f"need >=2 requests to measure consecutive overlap "
                 f"(got {len(requests)} from {src})")

    tok_bin = find_tokenize_bin()
    if tok_bin and not os.path.isfile(args.model):
        print(f"warn: model {args.model} missing; using fallback tokenizer",
              file=sys.stderr)
        tok_bin = None

    res = run(requests, args.model, tok_bin)
    res["source"] = src

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)

    pc = res["prefix_cache"]
    sc = res["semantic_cache"]
    print(f"requests={res['n_requests']}  pairs={res['n_consecutive_pairs']}  "
          f"tokenizer={res['tokenizer']}")
    print(f"PREFIX  mean_shared={pc['mean_shared_prefix_frac']*100:.1f}%  "
          f"median={pc['median_shared_prefix_frac']*100:.1f}%  "
          f"session_reuse={pc['session_reuse_frac']*100:.1f}% "
          f"({pc['cached_tokens']}/{res['total_prompt_tokens']} tok)")
    print(f"        buckets={pc['distribution_buckets']}")
    print(f"SEMANTIC near-dup (cosine)={sc['near_dup_rate_cosine']}")
    print(f"         verified        ={sc['verified_near_dup_rate_cosine']}")
    print(f"VERDICT (energy proxy) = {res['verdict']}  [{args.out}]")


if __name__ == "__main__":
    main()
