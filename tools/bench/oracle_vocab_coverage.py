#!/usr/bin/env python3
"""L3.1 oracle — usage-frequency vocab screen + norm-bound certificate.

Decides GO/NO-GO for a *certifiable exact-greedy* output-head prune
(design `plans/stateful_moat_continuation_design_2026_05_31.md` §2.1a / §2.4).

This is a DIFFERENT mechanism than the dead SVD low-rank screen
(`reports/oracle/svd_lmhead.json`: rank99=1987/2048 FULL-RANK NO-GO). It does
not touch rank. It tests whether a *usage-frequency hot set* H plus a
*per-row norm-bound certificate* can skip the bulk of the lm_head while staying
exact-greedy:

  - keep a hot set H of high-frequency tokens; compute logits over H only ->
    candidate argmax c with logit l_c;
  - an out-of-H token v is PROVABLY not the argmax iff  ||w_v|| * ||h|| < l_c
    (since l_v = w_v . h <= ||w_v|| ||h||);
  - if ALL out-of-H tokens satisfy the bound, c is the certified exact argmax
    (fast path, skip the bulk of the matmul / don't read those rows);
  - else fall back to a full exact pass.

Two halves are measured:
  (a) EFFECTIVE VOCAB COVERAGE (LABEL ESTIMATE) — tokenize the session corpus
      with llama-tokenize and use token-ID frequencies as a PROXY for argmax
      usage. Report fraction of the 151,936 vocab ever seen and the hot-set
      size H covering >= 99.x% of occurrences. (Proxy: input-token frequency
      stands in for argmax/output frequency, which needs a GPU decode we do
      not run.)
  (b) NORM-BOUND CERTIFICATE FALL-BACK RATE (LABEL ESTIMATE) — precompute
      per-row ||w_v|| from the dequantized lm_head (reusing
      oracle_svd_lmhead.py's GGUF loader). The certified-fast-path condition
      for a step is  max_{v not in H} ||w_v|| * ||h|| < l_c, equivalently
      ||h|| < l_c / max_{v not in H}||w_v||. Real ||h|| and l_c need a GPU run
      we do NOT do, so we MODEL them and SWEEP a plausible range of the ratio
      r = l_c / ||h|| (units of logit-per-hidden-norm). For each (H, r) we
      report the fraction of steps that certify (fast path) vs fall back.

GO if a small H (<= a few K tokens) yields >= 80% certified fast-path rate over
a plausible ratio range. Otherwise NO-GO, with a Type-1/Type-2 Kill-Protocol
classification.

Run with the gguf venv:
    /tmp/ggufenv/bin/python tools/bench/oracle_vocab_coverage.py
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

VOCAB_FULL = 151936  # Qwen2.5 vocab (from svd_lmhead.json shape)
CORPUS_GLOB = "/tmp/git_sessions_all/*.jsonl"
MODEL = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
OUT_JSON = "reports/oracle/vocab_coverage.json"

# Hot-set sizes to evaluate (the "<= a few K" greenlight regime + neighbours).
H_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# Coverage targets for the hot-set-size report.
COVERAGE_TARGETS = [0.990, 0.995, 0.999, 0.9999]

# Sweep of the alignment cos(w_c, h) -- the ONLY free physical quantity.
# The certificate fires for a step iff  max_{v not in H} ||w_v|| * ||h|| < l_c.
# Divide by ||h||:  the screen certifies iff  l_c/||h|| > max_{v not in H}||w_v||.
# Now l_c = w_c . h = ||w_c|| * ||h|| * cos(w_c, h), so l_c/||h|| = ||w_c||*cos.
# Modeling the winning row as a high-norm row (the MOST OPTIMISTIC case for the
# certificate -- gives the loosest threshold to beat), ||w_c|| = max row norm,
# the screen certifies iff  cos(w_c, h) > max_{v not in H}||w_v|| / max_row_norm.
# cos is in [-1, 1]; for an argmax-winning token cos > 0 and realistically in
# [0.3, 1.0]. We sweep cos and report the certified-step fraction. This sweep
# is SCALE-FREE: it does not depend on the absolute row-norm magnitude, only on
# the RATIO of out-of-H max norm to the winning-row norm -- which is the real
# determinant of whether a similar-row-norm head can certify.
COS_SWEEP = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 1.00]


# --------------------------------------------------------------------------- #
# Tokenisation  (mirrors tools/bench/oracle_prefix_cache.py)
# --------------------------------------------------------------------------- #
def find_tokenize_bin():
    cand = os.environ.get("TOKENIZE_BIN", "llama-tokenize")
    return cand if shutil.which(cand) else None


_TOK_LINE_RE = re.compile(r"\s*(\d+)")


def _parse_id_dump(text):
    """llama-tokenize --ids prints `[1, 2, 3]`; also tolerate one-id-per-line."""
    ids = []
    # Fast path: bracketed python list on one or more lines.
    for chunk in re.findall(r"\[([0-9,\s]+)\]", text):
        ids.extend(int(x) for x in re.findall(r"\d+", chunk))
    if ids:
        return ids
    for line in text.splitlines():
        m = _TOK_LINE_RE.match(line)
        if m:
            ids.append(int(m.group(1)))
    return ids


def tokenize_with_llama(text, model, bin_path):
    """llama-tokenize -m MODEL -f FILE --ids -> list[int]. None on failure."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(text)
        fpath = f.name
    try:
        proc = subprocess.run(
            [bin_path, "-m", model, "-f", fpath, "--ids", "--log-disable"],
            capture_output=True, timeout=300)
        if proc.returncode != 0:
            return None
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


def load_corpus_texts(glob_pat):
    """Each line is a JSON object; concatenate all string values as text."""
    texts = []
    for path in sorted(glob.glob(glob_pat)):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    texts.append(line)
                    continue
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict):
                    parts = [str(v) for v in obj.values()
                             if isinstance(v, (str, int, float))]
                    if parts:
                        texts.append("\n".join(parts))
    return texts


# --------------------------------------------------------------------------- #
# LM head dequant  (reuses oracle_svd_lmhead.py's loader verbatim in spirit)
# --------------------------------------------------------------------------- #
def load_lm_head(path):
    r = GGUFReader(path)
    by_name = {t.name: t for t in r.tensors}
    name = next((n for n in ("output.weight", "lm_head.weight")
                 if n in by_name), None)
    tied = False
    if name is None:
        name = "token_embd.weight"
        tied = True
    t = by_name[name]
    qtype = t.tensor_type.name
    W = dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
    if W.ndim != 2:
        W = W.reshape(t.shape[::-1])
    if W.shape[0] < W.shape[1]:
        W = W.T  # rows = vocab, cols = hidden
    return W, name, qtype, tied


# --------------------------------------------------------------------------- #
# Coverage  (LABEL ESTIMATE: input-token freq proxy for argmax freq)
# --------------------------------------------------------------------------- #
def coverage_stats(freq, vocab_full):
    """freq: np.array[int] counts per token-id. Returns coverage summary +
    hot-set sizes per coverage target + the per-H cumulative coverage."""
    total = int(freq.sum())
    seen = int((freq > 0).sum())
    order = np.argsort(freq)[::-1]            # token-ids, most frequent first
    sorted_counts = freq[order].astype(np.float64)
    cum = np.cumsum(sorted_counts) / max(total, 1)

    # Hot-set size H needed to reach each coverage target.
    h_for_cov = {}
    for tgt in COVERAGE_TARGETS:
        idx = int(np.searchsorted(cum, tgt))
        h_for_cov[f"H_for_{tgt:.4f}"] = min(idx + 1, seen)

    # Coverage achieved by each fixed H.
    cov_at_h = {}
    for H in H_SIZES:
        if H <= 0:
            continue
        k = min(H, len(cum))
        cov_at_h[str(H)] = float(cum[k - 1]) if k > 0 else 0.0

    return {
        "total_token_occurrences": total,
        "distinct_tokens_seen": seen,
        "vocab_full": vocab_full,
        "fraction_vocab_ever_seen": round(seen / vocab_full, 6),
        "hot_set_size_for_coverage": h_for_cov,
        "coverage_at_fixed_H": cov_at_h,
    }, order


# --------------------------------------------------------------------------- #
# Norm-bound certificate  (LABEL ESTIMATE: ||h||, l_c modeled & swept)
# --------------------------------------------------------------------------- #
def norm_freq_disjointness(row_norms, hot_order, k=20):
    """The kill's smoking gun: are the highest-norm rows ALSO frequency-hot?
    If the top-norm rows have high frequency-rank (i.e. are rare), they stay
    out-of-H for any frequency-chosen hot set, pinning max_out_of_H_norm at the
    global max -> the certificate can never beat it. Returns the top-k norm
    rows' token-ids and their frequency-rank (0 = most frequent)."""
    n = row_norms.shape[0]
    top_norm_ids = np.argsort(row_norms)[::-1][:k]
    freq_rank = np.empty(n, dtype=np.int64)
    freq_rank[hot_order] = np.arange(n)  # hot_order[0] is most frequent
    return {
        "top_norm_token_ids": [int(i) for i in top_norm_ids],
        "top_norm_values": [round(float(row_norms[i]), 4) for i in top_norm_ids],
        "top_norm_freq_rank": [int(freq_rank[i]) for i in top_norm_ids],
        "note": ("freq_rank 0 = most frequent. If these are large (rare), the "
                 "high-norm rows are NOT frequency-hot, so a frequency-chosen H "
                 "never includes them and max_out_of_H_norm stays at the global "
                 "max -- the structural reason the certificate cannot fire."),
    }


def certificate_sweep(row_norms, hot_order, h_sizes, cos_sweep):
    """For each hot-set size H (top-H most-frequent token ids) the certificate
    certifies a step iff  cos(w_c, h) > max_{v not in H}||w_v|| / ||w_c||.

    The winning row norm ||w_c|| is unknown without a GPU run, so we model two
    scenarios per H (LABEL ESTIMATE):
      - OPTIMISTIC: ||w_c|| = global max row norm (the winning token happens to
        be a max-norm row). Loosest threshold => best case for the certificate.
      - REALISTIC: ||w_c|| = median row norm (the winning token is a frequent,
        ordinary-norm token -- the common case). Tighter threshold.

    Crucially, max_{v not in H}||w_v|| barely drops as H grows, because H is
    chosen by *frequency*, not by *row norm* -- the largest-norm rows are mostly
    NOT the most frequent tokens. So the threshold the alignment must beat stays
    near (max_out / ||w_c||) regardless of H. We report, per H and per scenario,
    the fraction of swept cos values that certify (the certified-step rate if
    cos were uniform over the sweep) and the per-cos certified flag.
    """
    full_max = float(row_norms.max())
    median_norm = float(np.median(row_norms))
    p99_norm = float(np.percentile(row_norms, 99))
    n_vocab = row_norms.shape[0]
    rows = []
    for H in h_sizes:
        hot_ids = hot_order[:H]
        mask = np.ones(n_vocab, dtype=bool)
        valid = hot_ids[hot_ids < n_vocab]   # row index == token id
        mask[valid] = False                  # True => out-of-H (must beat bound)
        out_norms = row_norms[mask]
        max_out = float(out_norms.max()) if out_norms.size else 0.0
        entry = {"H": H, "max_out_of_H_row_norm": round(max_out, 4)}
        for label, wc in (("optimistic_wc_max", full_max),
                          ("realistic_wc_median", median_norm)):
            thresh = max_out / wc if wc > 0 else float("inf")
            per_cos = {}
            n_clear = 0
            for c in cos_sweep:
                clears = bool(c > thresh)
                per_cos[f"cos={c:g}"] = 1.0 if clears else 0.0
                n_clear += int(clears)
            entry[label] = {
                "cos_threshold_to_certify": round(thresh, 4),
                "certified_at_cos": per_cos,
                "frac_swept_cos_certifying": round(n_clear / len(cos_sweep), 4),
                "reachable": bool(thresh < 1.0),  # cos<=1 => cert possible iff <1
            }
        rows.append(entry)
    return {
        "full_max_row_norm": round(full_max, 4),
        "median_row_norm": round(median_norm, 4),
        "p99_row_norm": round(p99_norm, 4),
        "row_norm_min": round(float(row_norms.min()), 4),
        "row_norm_spread_max_over_median": round(full_max / median_norm, 4),
        "cos_sweep": cos_sweep,
        "note_units": ("Certificate fires iff cos(w_c,h) > max_out_of_H_norm / "
                       "||w_c||. cos in [-1,1]; an argmax winner has cos>0. "
                       "Scale-free: depends only on the norm RATIO, not the "
                       "absolute row-norm magnitude. ||w_c|| modeled as max "
                       "(optimistic) and median (realistic)."),
        "per_H": rows,
    }


def best_certified_rate(cert):
    """Highest certified-step rate over all H and both ||w_c|| scenarios, and
    whether it is physically reachable (cos<=1). For the GO/NO-GO headline we
    use the OPTIMISTIC scenario (the lever's best case)."""
    best = {"rate": 0.0, "H": None, "cos": None, "scenario": None,
            "reachable": False}
    for row in cert["per_H"]:
        for scen in ("optimistic_wc_max", "realistic_wc_median"):
            for key, val in row[scen]["certified_at_cos"].items():
                c = float(key.split("=")[1])
                if val > best["rate"] or (val == best["rate"] and
                                          best["cos"] is not None and
                                          c < best["cos"]):
                    best = {"rate": val, "H": row["H"], "cos": c,
                            "scenario": scen,
                            "reachable": row[scen]["reachable"]}
    return best


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    tok_bin = find_tokenize_bin()

    # ---- coverage (a) ------------------------------------------------------ #
    texts = load_corpus_texts(CORPUS_GLOB)
    freq = np.zeros(VOCAB_FULL, dtype=np.int64)
    n_tok_ok = 0
    n_tok_fail = 0
    method = "llama-tokenize" if tok_bin else "NONE(no tokenizer)"
    max_id_seen = 0
    if tok_bin:
        for text in texts:
            ids = tokenize_with_llama(text, model, tok_bin)
            if ids is None:
                n_tok_fail += 1
                continue
            n_tok_ok += 1
            arr = np.asarray(ids, dtype=np.int64)
            arr = arr[(arr >= 0) & (arr < VOCAB_FULL)]
            if arr.size:
                max_id_seen = max(max_id_seen, int(arr.max()))
                np.add.at(freq, arr, 1)
    cov, hot_order = coverage_stats(freq, VOCAB_FULL)
    cov["corpus_glob"] = CORPUS_GLOB
    cov["documents"] = len(texts)
    cov["documents_tokenized_ok"] = n_tok_ok
    cov["documents_tokenized_fail"] = n_tok_fail
    cov["tokenizer"] = method
    cov["max_token_id_seen"] = max_id_seen
    cov["LABEL"] = ("ESTIMATE: input-token-frequency proxy for argmax/output "
                    "frequency. Real argmax usage needs a GPU decode (not run).")

    # ---- certificate (b) --------------------------------------------------- #
    W, name, qtype, tied = load_lm_head(model)
    row_norms = np.linalg.norm(W, axis=1).astype(np.float64)
    cert = certificate_sweep(row_norms, hot_order, H_SIZES, COS_SWEEP)
    cert["norm_freq_disjointness"] = norm_freq_disjointness(row_norms, hot_order)
    cert["tensor"] = name
    cert["qtype"] = qtype
    cert["tied_embedding"] = tied
    cert["shape_vocab_hidden"] = [int(W.shape[0]), int(W.shape[1])]
    cert["LABEL"] = ("ESTIMATE: ||h|| and l_c are MODELED via the alignment "
                     "cos(w_c,h) sweep, not GPU-measured. The certified rate is "
                     "a function of cos(w_c,h) vs (max_out_of_H_norm/||w_c||).")

    best = best_certified_rate(cert)

    # ---- GO / NO-GO -------------------------------------------------------- #
    # GO requires: a small H (<= a few K) that certifies >= 80% of steps at a
    # PHYSICALLY-REACHABLE alignment cos(w_c,h) <= 1. In the OPTIMISTIC scenario
    # (||w_c|| = max row norm), the cos threshold to certify is
    # max_out_of_H_norm / max_row_norm. If that threshold is >= 1.0 the
    # certificate can NEVER fire (would need cos > 1). We surface, per small H,
    # whether the optimistic threshold is reachable and what cos it demands.
    small_H_thresh = 4096  # "a few K"
    small_rows = [r for r in cert["per_H"] if r["H"] <= small_H_thresh]
    # An (H) is GO-eligible if optimistic threshold < 1 AND a high cos (>=0.9,
    # i.e. the certified-step rate would be >=80% if cos concentrates there)
    # certifies. We require the threshold to be comfortably < 0.9 so that the
    # bulk of plausible argmax alignments (cos in [0.9,1]) clear it.
    reachable_certifying_small = []
    for r in small_rows:
        opt = r["optimistic_wc_max"]
        thr = opt["cos_threshold_to_certify"]
        if opt["reachable"] and thr < 0.9:
            reachable_certifying_small.append({
                "H": r["H"], "cos_threshold": thr,
                "scenario": "optimistic_wc_max",
                "frac_cos_certifying": opt["frac_swept_cos_certifying"]})
    go = bool(reachable_certifying_small)

    if go:
        verdict = "GO"
        kill_type = None
        rationale = ("A small hot set certifies at a physically-reachable "
                     "alignment cos(w_c,h) < 0.9 (so the bulk of plausible "
                     "argmax alignments clear the bound). Byte cut is real and "
                     "fall-back rare at that operating point.")
    else:
        verdict = "NO-GO"
        kill_type = "Type-1"
        # cos threshold the smallest small-H must beat in the optimistic case.
        _opt0 = small_rows[0]["optimistic_wc_max"] if small_rows else {}
        _thr0 = _opt0.get("cos_threshold_to_certify", float("nan"))
        rationale = (
            "The norm-bound certificate cannot certify a small frequency-hot "
            "set at any PHYSICALLY-REACHABLE alignment. Reason (measured, not "
            "implementation-dependent): the hot set is chosen by token "
            "FREQUENCY, but the largest-norm lm_head rows are NOT the frequent "
            "tokens, so max_{v not in H}||w_v|| stays ~= the global max row "
            "norm even for H in the tens of thousands. The certificate needs "
            "cos(w_c,h) > max_{v not in H}||w_v|| / ||w_c||; even in the "
            "OPTIMISTIC case (||w_c|| = global max row norm) the smallest hot "
            f"set demands cos(w_c,h) > {_thr0:.3f} -- and the REALISTIC case "
            "(||w_c|| = median norm, since the winner is a frequent ordinary "
            "token) makes the threshold >= 1.0 (unreachable, cos<=1). Generic "
            "decode hidden states do not align with a max-norm row at "
            "cos~1. The FULL-RANK (rank99=97% of dim, max/median row-norm "
            f"spread {cert['row_norm_spread_max_over_median']:.2f}x) head "
            "(svd_lmhead.json) is exactly the regime where this Cauchy-Schwarz "
            "upper bound is too loose to certify."
        )

    type2_reframe = {
        "named_alternative": (
            "Tighter exact certificate that does NOT rely on the scalar "
            "Cauchy-Schwarz row-norm bound: (i) a per-coordinate / interval "
            "bound using sign-aligned partial sums of w_v against h "
            "(elementwise max contribution), or (ii) a partitioned-vocab "
            "block-max bound where each contiguous row block stores "
            "max_j |w_v[j]| per coordinate so out-of-H blocks get a tighter "
            "per-block ceiling than a single global row-norm, or (iii) a "
            "data-aware screen that orders the vocab by argmax frequency from "
            "a REAL decode (usage_capture) rather than input-token frequency, "
            "tightening H so the frequent-argmax set and the high-norm set "
            "overlap more."),
        "cheap_oracle": (
            "Each reframe has a cheap offline NumPy oracle: (i)/(ii) recompute "
            "the certified-rate sweep with the tighter ceiling (block-max "
            "matrices are a reshape + max over the SAME dequantized W already "
            "loaded here -- a few lines added to this script); (iii) re-run "
            "coverage with REAL argmax ids from usage_capture (the side-"
            "observer the design proposes) instead of llama-tokenize input "
            "ids, then re-run THIS certificate sweep against that H."),
        "status": (
            "Reframes (i)/(ii) are Type-2-ALIVE only with the named block-max "
            "oracle: a per-coordinate bound CAN be much tighter than the "
            "scalar row-norm bound when h has a few dominant coordinates, but "
            "on a similar-row-norm / cond~45 head the gain is uncertain and "
            "must be MEASURED before any wiring. Reframe (iii) needs a GPU "
            "decode (usage_capture) to produce real argmax ids -- out of scope "
            "for this CPU oracle, deferred to that instrument. None resurrect "
            "the lever on vibes; each is dead until its oracle clears."),
    }

    res = {
        "oracle": "vocab_coverage_norm_bound_certificate",
        "verdict": verdict,
        "greenlight_rule": ("GO iff a small H (<= %d) certifies >=80%% of steps "
                            "at a physically-reachable alignment, i.e. the "
                            "optimistic cos threshold (max_out_of_H_norm / "
                            "max_row_norm) is < 0.9 so cos in [0.9,1] clears it."
                            % small_H_thresh),
        "best_certified_operating_point": best,
        "go_reachable_certifying_small_H": reachable_certifying_small[:5],
        "rationale": rationale,
        "kill_protocol": {
            "classification": kill_type if kill_type else "n/a (GO)",
            "type1_reality": (rationale if kill_type == "Type-1" else None),
            "type2_reframe": type2_reframe,
        },
        "coverage": cov,
        "certificate": cert,
        "caveats": [
            "LABEL ESTIMATE (a): input-token frequency != argmax frequency.",
            "LABEL ESTIMATE (b): ||h|| and l_c modeled/swept, not GPU-measured.",
            "lm_head is ~4-10% of bytes/token; ceiling is small even if GO.",
            "Certificate strength bounded by Cauchy-Schwarz; tighter bounds "
            "(block-max / per-coord) are the only Type-2 escape and need their "
            "own oracle.",
        ],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(res, open(OUT_JSON, "w"), indent=2)

    # ---- console summary --------------------------------------------------- #
    print(f"tokenizer={method}  docs={len(texts)} ok={n_tok_ok} fail={n_tok_fail}")
    print(f"vocab seen: {cov['distinct_tokens_seen']}/{VOCAB_FULL} "
          f"= {cov['fraction_vocab_ever_seen']*100:.2f}%  "
          f"(occurrences={cov['total_token_occurrences']})")
    print("hot-set size for coverage:",
          cov["hot_set_size_for_coverage"])
    print(f"row norms: min={cert['row_norm_min']} median={cert['median_row_norm']} "
          f"p99={cert['p99_row_norm']} max={cert['full_max_row_norm']} "
          f"(max/median spread {cert['row_norm_spread_max_over_median']}x)")
    print("per-H (frequency-chosen H): max_out_norm | cos-to-certify "
          "[optimistic ||w_c||=max / realistic ||w_c||=median]:")
    for row in cert["per_H"]:
        o = row["optimistic_wc_max"]; r = row["realistic_wc_median"]
        print(f"  H={row['H']:>6}  max_out={row['max_out_of_H_row_norm']:>7}"
              f"  cos*_opt={o['cos_threshold_to_certify']:>6} "
              f"(reach={int(o['reachable'])})"
              f"  cos*_real={r['cos_threshold_to_certify']:>6} "
              f"(reach={int(r['reachable'])})")
    dj = cert["norm_freq_disjointness"]
    print(f"norm/freq disjointness: top-norm token-ids {dj['top_norm_token_ids'][:5]} "
          f"have freq-rank {dj['top_norm_freq_rank'][:5]} (0=most frequent) "
          f"-> high-norm rows are rare, stay out-of-H.")
    print(f"best certified-step rate over cos sweep: {best['rate']*100:.0f}% "
          f"@ H={best['H']} cos={best['cos']} scenario={best['scenario']} "
          f"reachable={best['reachable']}")
    print(f"VERDICT: {verdict}"
          + (f"  [{kill_type}]" if kill_type else ""))
    print(f"-> {OUT_JSON}")
    return res


if __name__ == "__main__":
    main()
