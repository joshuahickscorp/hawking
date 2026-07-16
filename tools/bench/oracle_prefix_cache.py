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

Tokenisation shells out to $TOKENIZE_BIN (default llama-tokenize) against the
Qwen2.5-3B gguf. If that binary is unavailable we fall back to a
whitespace/byte tokenizer so the tool is never blocked (the fallback is
clearly flagged in the output).

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
import glob as _glob
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

# --- L1.2 SEMANTIC-UPLIFT sweep (design doc §1.6) -------------------------- #
# tau_sem: embedding-cosine retrieval threshold for nominating a prior context.
SEM_UPLIFT_TAUS = (0.95, 0.90, 0.80, 0.70)
# MIN_REUSE_TOKENS: a semantic hit only "verify-confirms" if its exact common
# prefix is at least this long (the design's correctness gate / precise retrieval).
SEM_UPLIFT_MIN_REUSE = (16, 32, 64)
# Greenlight: semantic-augmented reuse fraction must beat exact-only by this many
# percentage points at a tau with >= this verify-confirm rate.
UPLIFT_GATE_PTS = 10.0           # ~10 percentage points
VERIFY_CONFIRM_GATE = 0.95       # >=95% of semantic hits exact-confirm >= MIN_REUSE


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


# --------------------------------------------------------------------------- #
# L1.2 SEMANTIC-UPLIFT oracle (design doc §1.6)
# --------------------------------------------------------------------------- #
def tokenize_session(requests, model, tok_bin):
    """Tokenize a session's requests -> (list[list[int]], method, used_fallback)."""
    tokenized = []
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
    if tok_bin and used_fallback:
        method = "mixed(llama-tokenize + fallback on failures)"
    elif tok_bin:
        method = "llama-tokenize"
    else:
        method = "fallback(ws/byte)"
    return tokenized, method, used_fallback


def semantic_uplift_session(tokenized, taus, min_reuse_set):
    """Incremental reuse the SEMANTIC tier buys OVER the exact (consecutive) tier.

    For each request i (i>=1):
      (a) exact-consecutive reuse  = common_prefix_len(req[i-1], req[i])
          -- exactly what the shipped default-on prefix cache already gets.
      (b) best-prior SEMANTIC reuse(tau) = max over prior j<i with
          cosine(vec[i],vec[j]) >= tau of common_prefix_len(req[j], req[i])
          -- the longest exact common prefix against ANY semantically-near prior,
             not just the immediately preceding request.

    The session payoff is sum(b)-sum(a) over a session as a fraction of total
    prompt tokens. Also tracks, per tau, the VERIFY-CONFIRM rate: of requests
    that had >=1 semantic candidate (a prior j passing tau), the fraction whose
    best exact-common-prefix >= MIN_REUSE_TOKENS (precise retrieval, not noisy).

    Returns a dict keyed by tau -> {per MIN_REUSE_TOKENS metrics}, plus the
    shared exact-only baseline for the session.
    """
    n = len(tokenized)
    total_tokens = sum(len(t) for t in tokenized)
    vecs = [hashed_bigram_vec(t) for t in tokenized]

    # exact-consecutive reuse tokens per request i (the shipped cache's reuse).
    exact_reuse = [0] * n
    for i in range(1, n):
        exact_reuse[i] = common_prefix_len(tokenized[i - 1], tokenized[i])
    exact_total = sum(exact_reuse)
    exact_frac = exact_total / total_tokens if total_tokens else 0.0

    # Precompute, for each (i, j<i): cosine and exact common prefix len. The
    # prefix len is independent of tau; only the candidate set (cosine>=tau)
    # changes across the tau sweep, so compute the pairwise tables once.
    # best_prefix_at_cos[i] = list of (cos_ij, cpl_ij) for j<i.
    pair_tables = [[] for _ in range(n)]
    for i in range(1, n):
        for j in range(i):
            c = cosine(vecs[i], vecs[j])
            cpl = common_prefix_len(tokenized[j], tokenized[i])
            pair_tables[i].append((c, cpl))

    out = {
        "n_requests": n,
        "total_prompt_tokens": total_tokens,
        "exact_only_reuse_tokens": exact_total,
        "exact_only_reuse_frac": round(exact_frac, 4),
        "by_tau": {},
    }
    for tau in taus:
        # best semantic prefix per request among candidates passing tau.
        sem_best_prefix = [0] * n          # best exact-common-prefix vs any near prior
        had_candidate = [False] * n        # >=1 prior passed tau (a semantic "hit")
        for i in range(1, n):
            best = 0
            cand = False
            for (c, cpl) in pair_tables[i]:
                if c >= tau:
                    cand = True
                    if cpl > best:
                        best = cpl
            sem_best_prefix[i] = best
            had_candidate[i] = cand
        n_candidates = sum(1 for x in had_candidate if x)

        tau_block = {"n_semantic_candidate_requests": n_candidates,
                     "by_min_reuse": {}}
        for mr in min_reuse_set:
            # The semantic tier only reuses a candidate whose exact common prefix
            # >= MIN_REUSE_TOKENS; otherwise it falls through (no reuse credited
            # beyond what exact-consecutive already gives). The augmented reuse
            # for request i = max(exact_reuse[i], sem_best_prefix[i] if it clears
            # the MIN_REUSE gate else exact_reuse[i]).
            aug = 0
            confirmed_hits = 0
            for i in range(n):
                sem_i = sem_best_prefix[i] if sem_best_prefix[i] >= mr else 0
                aug += max(exact_reuse[i], sem_i)
                if had_candidate[i] and sem_best_prefix[i] >= mr:
                    confirmed_hits += 1
            aug_frac = aug / total_tokens if total_tokens else 0.0
            delta_frac = aug_frac - exact_frac
            # verify-confirm rate: of requests with a semantic candidate, how
            # many had an exact common prefix >= MIN_REUSE (precise retrieval).
            verify_rate = (confirmed_hits / n_candidates) if n_candidates else 0.0
            tau_block["by_min_reuse"][str(mr)] = {
                "augmented_reuse_tokens": aug,
                "augmented_reuse_frac": round(aug_frac, 4),
                "incremental_reuse_frac": round(delta_frac, 4),
                "incremental_reuse_pts": round(delta_frac * 100, 2),
                "confirmed_semantic_hits": confirmed_hits,
                "verify_confirm_rate": round(verify_rate, 4),
            }
        out["by_tau"][str(tau)] = tau_block
    return out


def _spread(values):
    """min/median/max of a list of floats (rounded)."""
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0, "mean": 0.0, "n": 0}
    return {
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.fmean(values), 4),
        "n": len(values),
    }


def run_semantic_uplift(sessions, model, tok_bin,
                        taus=SEM_UPLIFT_TAUS, min_reuse_set=SEM_UPLIFT_MIN_REUSE):
    """Run the semantic-uplift measurement PER SESSION and aggregate.

    `sessions` is a list of (label, list[str] requests). Sessions with <2
    requests are skipped (no consecutive overlap to measure). Aggregates the
    incremental-reuse and verify-confirm distributions across sessions and
    decides GO/NO-GO per the design's greenlight (>=~10 pts uplift at a tau with
    >=95% verify-confirm).
    """
    per_session = []
    any_fallback = False
    method_seen = None
    for label, reqs in sessions:
        if len(reqs) < 2:
            per_session.append({"session": label, "skipped": "<2 requests",
                                "n_requests": len(reqs)})
            continue
        tokenized, method, used_fb = tokenize_session(reqs, model, tok_bin)
        any_fallback = any_fallback or used_fb
        method_seen = method
        s = semantic_uplift_session(tokenized, taus, min_reuse_set)
        s["session"] = label
        per_session.append(s)

    measured = [s for s in per_session if "exact_only_reuse_frac" in s]

    # Aggregate across sessions. For each (tau, min_reuse) cell, collect the
    # per-session incremental-reuse pts and verify-confirm rates -> spread.
    agg = {"by_tau": {}}
    exact_only_fracs = [s["exact_only_reuse_frac"] for s in measured]
    agg["exact_only_reuse_frac_spread"] = _spread(exact_only_fracs)
    for tau in taus:
        tb = {"by_min_reuse": {}}
        for mr in min_reuse_set:
            inc_pts, ver_rates, aug_fracs = [], [], []
            # verify-confirm aggregated as a weighted rate too (total confirmed /
            # total candidates across sessions) — robust to small per-session n.
            tot_conf, tot_cand = 0, 0
            for s in measured:
                cell = s["by_tau"][str(tau)]["by_min_reuse"][str(mr)]
                inc_pts.append(cell["incremental_reuse_pts"])
                aug_fracs.append(cell["augmented_reuse_frac"])
                # only sessions that actually had >=1 candidate inform the
                # per-session verify rate spread (else it's a vacuous 0/0->0).
                ncand = s["by_tau"][str(tau)]["n_semantic_candidate_requests"]
                if ncand:
                    ver_rates.append(cell["verify_confirm_rate"])
                tot_conf += cell["confirmed_semantic_hits"]
                tot_cand += ncand
            weighted_verify = (tot_conf / tot_cand) if tot_cand else 0.0
            tb["by_min_reuse"][str(mr)] = {
                "incremental_reuse_pts_spread": _spread(inc_pts),
                "augmented_reuse_frac_spread": _spread(aug_fracs),
                "verify_confirm_rate_spread": _spread(ver_rates),
                "verify_confirm_rate_weighted": round(weighted_verify, 4),
                "total_confirmed_hits": tot_conf,
                "total_semantic_candidate_requests": tot_cand,
            }
        agg["by_tau"][str(tau)] = tb

    # Verdict: GO if some (tau, min_reuse) has mean incremental >= UPLIFT_GATE_PTS
    # AND weighted verify-confirm >= VERIFY_CONFIRM_GATE.
    best = None
    for tau in taus:
        for mr in min_reuse_set:
            cell = agg["by_tau"][str(tau)]["by_min_reuse"][str(mr)]
            mean_pts = cell["incremental_reuse_pts_spread"]["mean"]
            ver = cell["verify_confirm_rate_weighted"]
            passes = (mean_pts >= UPLIFT_GATE_PTS and ver >= VERIFY_CONFIRM_GATE)
            cand = {"tau_sem": tau, "min_reuse_tokens": mr,
                    "mean_incremental_pts": mean_pts,
                    "median_incremental_pts": cell["incremental_reuse_pts_spread"]["median"],
                    "max_incremental_pts": cell["incremental_reuse_pts_spread"]["max"],
                    "verify_confirm_rate_weighted": ver,
                    "passes_gate": passes}
            # rank: prefer passing cells by mean pts; else track the max-uplift
            # cell for the report regardless of gate.
            if best is None:
                best = cand
            else:
                better = (
                    (cand["passes_gate"], cand["mean_incremental_pts"]) >
                    (best["passes_gate"], best["mean_incremental_pts"]))
                if better:
                    best = cand
    verdict = "GO" if (best and best["passes_gate"]) else "NO-GO"

    method = method_seen or ("fallback(ws/byte)" if tok_bin is None
                             else "llama-tokenize")
    return {
        "oracle": "prefix_cache_semantic_uplift",
        "lever": "§1.6 L1.2 semantic cache — incremental reuse over exact tier",
        "tokenizer": method,
        "used_fallback_tokenizer": any_fallback,
        "n_sessions": len(sessions),
        "n_sessions_measured": len(measured),
        "tau_sem_swept": list(taus),
        "min_reuse_tokens_swept": list(min_reuse_set),
        "gate": {"uplift_pts": UPLIFT_GATE_PTS,
                 "verify_confirm_rate": VERIFY_CONFIRM_GATE},
        "aggregate": agg,
        "best_cell": best,
        "verdict": verdict,
        "per_session": per_session,
        "note": (
            "incremental_reuse = (best-prior semantic reuse frac) - (exact-"
            "consecutive reuse frac). Exact-consecutive = common_prefix_len("
            "req[i-1],req[i]) (the SHIPPED default-on cache). Semantic = longest "
            "exact common prefix vs ANY prior j with embed-cosine>=tau_sem, "
            "credited only when that prefix >= MIN_REUSE_TOKENS (the verify "
            "gate). verify_confirm_rate = of requests with a semantic candidate, "
            "fraction whose best exact prefix >= MIN_REUSE_TOKENS. GO if mean "
            "incremental >= %.0f pts at a tau with weighted verify-confirm >= "
            "%.0f%%. ESTIMATE: git-history proxy for a real user; the production "
            "number comes from real session logs." % (
                UPLIFT_GATE_PTS, VERIFY_CONFIRM_GATE * 100)),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--texts", nargs="+", metavar="FILE",
                   help="transcript text files, in session order")
    g.add_argument("--jsonl", metavar="FILE",
                   help="JSONL of request objects, one request per line")
    g.add_argument("--sessions-glob", metavar="GLOB",
                   help="glob of per-session JSONL files (each file = one "
                        "session; each line = one request). Runs the L1.2 "
                        "SEMANTIC-UPLIFT oracle (§1.6): incremental reuse the "
                        "semantic tier buys over the exact tier, per session, "
                        "aggregated with spread. Writes the uplift JSON.")
    ap.add_argument("--split-on", default=None,
                    help="with --texts: split each file into a request "
                         "sequence on this delimiter (e.g. a turn marker)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="gguf for llama-tokenize (default %(default)s)")
    ap.add_argument("--out", default="reports/oracle/prefix_cache.json")
    args = ap.parse_args()

    tok_bin = find_tokenize_bin()
    if tok_bin and not os.path.isfile(args.model):
        print(f"warn: model {args.model} missing; using fallback tokenizer",
              file=sys.stderr)
        tok_bin = None

    # ---- L1.2 SEMANTIC-UPLIFT mode (multi-session) ---------------------- #
    if args.sessions_glob:
        paths = sorted(_glob.glob(args.sessions_glob))
        if not paths:
            sys.exit(f"no files matched --sessions-glob {args.sessions_glob!r}")
        sessions = []
        for p in paths:
            reqs = []
            for ln in open(p, encoding="utf-8", errors="replace"):
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
            sessions.append((os.path.basename(p), reqs))
        res = run_semantic_uplift(sessions, args.model, tok_bin)
        res["source"] = f"sessions-glob:{args.sessions_glob} ({len(paths)} files)"
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)

        eo = res["aggregate"]["exact_only_reuse_frac_spread"]
        print(f"SEMANTIC-UPLIFT oracle  sessions={res['n_sessions']} "
              f"(measured {res['n_sessions_measured']})  "
              f"tokenizer={res['tokenizer']}")
        print(f"exact-only reuse frac: min={eo['min']*100:.1f}% "
              f"median={eo['median']*100:.1f}% max={eo['max']*100:.1f}% "
              f"mean={eo['mean']*100:.1f}%")
        for tau in res["tau_sem_swept"]:
            for mr in res["min_reuse_tokens_swept"]:
                cell = res["aggregate"]["by_tau"][str(tau)]["by_min_reuse"][str(mr)]
                sp = cell["incremental_reuse_pts_spread"]
                print(f"  tau={tau} MIN_REUSE={mr:>2}: "
                      f"{sp['mean']:+.2f} pts mean "
                      f"(min{sp['min']:+.2f}/med{sp['median']:+.2f}/max{sp['max']:+.2f}) "
                      f"verify-confirm={cell['verify_confirm_rate_weighted']*100:.1f}%")
        b = res["best_cell"]
        if b:
            print(f"BEST: tau={b['tau_sem']} MIN_REUSE={b['min_reuse_tokens']} "
                  f"mean={b['mean_incremental_pts']:+.2f} pts "
                  f"verify={b['verify_confirm_rate_weighted']*100:.1f}% "
                  f"passes_gate={b['passes_gate']}")
        print(f"VERDICT = {res['verdict']}  "
              f"(gate: >={UPLIFT_GATE_PTS:.0f} pts @ "
              f">={VERIFY_CONFIRM_GATE*100:.0f}% verify-confirm)  [{args.out}]")
        return

    requests, src = load_requests(args)
    if len(requests) < 2:
        sys.exit(f"need >=2 requests to measure consecutive overlap "
                 f"(got {len(requests)} from {src})")

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
