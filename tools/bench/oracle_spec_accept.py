#!/usr/bin/env python3
"""Bible Stage-0 oracle A — n-gram / prompt-lookup (PLD) speculation acceptance.

Offline, lossless. Simulates PLD on a real CODE token stream: at each step,
match the current n-gram suffix against the earliest-emitted prefix, draft up
to K tokens from the prior continuation, accept the leading run that matches
the true stream. Mean accepted length τ = tokens emitted per verify forward.
Bible threshold: τ ≥ ~2.5 ⇒ n-gram/SAM speculation is a strong win on this
workload (the draft is a ~free CPU automaton, so τ is the speedup ceiling).

This proxies code-completion serving (prompt+generation are both code) on the
copy structure of real code. Input: a `llama-tokenize -f` dump (id -> 'piece').

------------------------------------------------------------------------------
L3.1 DRAFT-TUNING extension (`--sessions DIR`)  —  bible §8 L3.1 / design §2.4
------------------------------------------------------------------------------
The single-stream mode above measures GENERIC PLD (τ=1.43 on code,
reports/oracle/spec_accept.json). L3.1 asks a different question: does
warm-starting the draft index on a USER's OWN prior token stream lift τ above
the generic baseline (per-user specialization)? We proxy a "user" with a
git-history coding session: a sequence of turns iterating on a working set. We
split each session into a warm prior slice (seed the draft index) and a held-out
later slice (measure τ).

Honesty guard against double-counting the SHIPPED, default-on prefix cache
(L1.2): consecutive turns share a long exact prefix (re-sent files) that the
prefix cache already restores at prefill — so a warm draft index would "draft"
that prefix with near-perfect acceptance and INFLATE τ with reuse we already
bank elsewhere. We therefore report FOUR numbers per session:

  tau_warm_full   warm seed, count ALL held-out tokens          (handoff's literal
                                                                  ask; UPPER proxy)
  tau_cold_full   no history, count ALL held-out tokens          (matched control)
  tau_warm_suffix warm seed, count ONLY post-shared-prefix tokens (the part the
                                                                  prefix cache does
                                                                  NOT serve)
  tau_cold_suffix no history, count ONLY post-shared-prefix tokens

tau_warm_suffix is the ADDITIVE draft value — what user-history specialization
buys on top of the already-shipped prefix cache. The greenlight keys on it.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from statistics import median

from oracle_prefix_cache import common_prefix_len


def load_ids(path):
    ids = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\s*(\d+)", line)
        if m:
            ids.append(int(m.group(1)))
    return ids


def simulate(ids, n_match, K, start=0, seed_warm=True, count_mask=None):
    """PLD over `ids`. The index maps an n-gram (ids[t-n_match:t]) to its most
    recent continuation index t (draft = ids[t:t+K]); registered AFTER the
    query at t so a gram never self-matches.

    start       — first position eligible to be a measured step.
    seed_warm   — if True, pre-seed the index over [n_match, start) (the warm
                  prior slice = user history). If False, the index starts EMPTY
                  at `start` (cold: no cross-slice history).
    count_mask  — optional bool list aligned to `ids`; a step at position i
                  contributes to τ only if count_mask[i] is True. EVERY emitted
                  position still registers its gram regardless of the mask, so
                  the index always reflects the full context (only the τ
                  denominator is restricted). A step is attributed by its START
                  position i.
    """
    idx = {}
    if seed_warm:
        for t in range(n_match, start):
            idx[tuple(ids[t - n_match:t])] = t
    N = len(ids)
    i = max(start, n_match)
    steps = emitted = 0
    acc = defaultdict(int)
    while i < N:
        counted = count_mask is None or (i < len(count_mask) and count_mask[i])
        a = 0
        key = tuple(ids[i - n_match:i])
        j = idx.get(key)
        if j is not None and j < i:
            while a < K and i + a < N and j + a < N and ids[j + a] == ids[i + a]:
                a += 1
        adv = a + 1
        end = min(i + adv, N)
        for t in range(i, end):
            if t >= n_match:
                idx[tuple(ids[t - n_match:t])] = t
        if counted:
            steps += 1
            emitted += end - i
            acc[a] += 1
        i = end
    tau = emitted / steps if steps else 0.0
    return {
        "n_match": n_match, "K": K, "steps": steps, "emitted": emitted,
        "mean_accepted_len": round(tau, 3),
        "hit_rate": round(sum(c for a, c in acc.items() if a > 0) / steps, 3) if steps else 0,
        "acc_hist": {str(a): acc[a] for a in sorted(acc)},
    }


# --------------------------------------------------------------------------- #
# Generic single-stream mode (unchanged behaviour; produces the τ=1.43 baseline)
# --------------------------------------------------------------------------- #
def run_single_stream(ids, out):
    half = len(ids) // 2
    grid = []
    for n_match in (2, 3):
        for K in (8, 16):
            full = simulate(ids, n_match, K, start=0)
            warm = simulate(ids, n_match, K, start=half)
            full["warm_half_mean_accepted_len"] = warm["mean_accepted_len"]
            full["warm_half_hit_rate"] = warm["hit_rate"]
            grid.append(full)
    best = max(grid, key=lambda r: r["warm_half_mean_accepted_len"])
    verdict = ("GO" if best["warm_half_mean_accepted_len"] >= 2.5
               else "MARGINAL" if best["warm_half_mean_accepted_len"] >= 1.6
               else "NO-GO")
    out_obj = {
        "oracle": "spec_accept_pld_ngram",
        "tokens": len(ids),
        "threshold_tau": 2.5,
        "best": {k: best[k] for k in
                 ("n_match", "K", "mean_accepted_len",
                  "warm_half_mean_accepted_len", "warm_half_hit_rate")},
        "verdict": verdict,
        "grid": grid,
        "note": ("PLD on real code; τ=tokens/forward. Proxies code-completion "
                 "(prompt+gen both code). Draft is ~free CPU automaton, so τ is "
                 "the speedup ceiling. GO≥2.5, MARGINAL≥1.6, else NO-GO."),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(out_obj, open(out, "w"), indent=2)
    print(f"tokens={len(ids)}  verdict={verdict}")
    for r in grid:
        print(f"  n={r['n_match']} K={r['K']:2d}: τ_full={r['mean_accepted_len']:.2f} "
              f"τ_warm={r['warm_half_mean_accepted_len']:.2f} "
              f"hit_warm={r['warm_half_hit_rate']:.2f}")
    print(f"BEST warm τ={best['warm_half_mean_accepted_len']:.2f} "
          f"(n={best['n_match']},K={best['K']}) -> {verdict}  [{out}]")


# --------------------------------------------------------------------------- #
# L3.1 per-session warm-start mode
# --------------------------------------------------------------------------- #
GENERIC_TAU = 1.43          # the reports/oracle/spec_accept.json baseline to beat
GO_TAU = 1.8                # "even τ≥1.8 is a real per-user win" (design §2.4)
MARGINAL_TAU = 1.6
SPECIALIZATION_MARGIN = 0.20  # warm_suffix must beat cold_suffix by this to be
#                               genuine specialization (not substrate luck)


def tokenize_llama(text, model, bin_path):
    """llama-tokenize -m MODEL -f FILE -> list[int] (None on failure)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as f:
        f.write(text)
        fpath = f.name
    try:
        proc = subprocess.run([bin_path, "-m", model, "-f", fpath],
                              capture_output=True, timeout=300)
        if proc.returncode != 0:
            return None
        ids = []
        for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
            m = re.match(r"\s*(\d+)", line)
            if m:
                ids.append(int(m.group(1)))
        return ids or None
    except (subprocess.TimeoutExpired, OSError):
        return None
    finally:
        try:
            os.unlink(fpath)
        except OSError:
            pass


def load_session_turns(path):
    turns = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, str):
            turns.append(obj)
            continue
        for key in ("request", "prompt", "text", "content"):
            if key in obj and isinstance(obj[key], str):
                turns.append(obj[key])
                break
    return turns


def analyze_session(turns_ids, n_match, K):
    """turns_ids: list[list[int]] in session order. Split first-half warm /
    second-half held-out; return the 4 τ variants + aux."""
    n_turns = len(turns_ids)
    W = n_turns // 2
    if W < 1 or n_turns - W < 1:
        return None
    stream = []
    spans = []
    for t in turns_ids:
        spans.append((len(stream), len(stream) + len(t)))
        stream.extend(t)
    seed_end = spans[W][0]
    # suffix mask: held-out positions NOT in the leading exact-common-prefix
    # with the previous turn (= the part the default-on prefix cache restores).
    mask = [False] * len(stream)
    held_tokens = 0
    prefix_served = 0
    for ti in range(W, n_turns):
        s, e = spans[ti]
        p = common_prefix_len(turns_ids[ti - 1], turns_ids[ti])
        p = min(p, e - s)
        held_tokens += e - s
        prefix_served += p
        for pos in range(s + p, e):
            mask[pos] = True
    wf = simulate(stream, n_match, K, start=seed_end, seed_warm=True)
    cf = simulate(stream, n_match, K, start=seed_end, seed_warm=False)
    ws = simulate(stream, n_match, K, start=seed_end, seed_warm=True, count_mask=mask)
    cs = simulate(stream, n_match, K, start=seed_end, seed_warm=False, count_mask=mask)
    return {
        "turns": n_turns, "warm_turns": W, "held_turns": n_turns - W,
        "held_tokens": held_tokens,
        "prefix_cache_served_frac": round(prefix_served / held_tokens, 3) if held_tokens else 0.0,
        "tau_warm_full": wf["mean_accepted_len"],
        "tau_cold_full": cf["mean_accepted_len"],
        "tau_warm_suffix": ws["mean_accepted_len"],
        "tau_cold_suffix": cs["mean_accepted_len"],
        "warm_suffix_hit_rate": ws["hit_rate"],
        # raw step/emitted for pooled aggregation
        "_pool": {
            "warm_full": (wf["steps"], wf["emitted"]),
            "cold_full": (cf["steps"], cf["emitted"]),
            "warm_suffix": (ws["steps"], ws["emitted"]),
            "cold_suffix": (cs["steps"], cs["emitted"]),
        },
    }


def run_sessions(session_files, model, tok_bin, n_match, K, min_turns, out):
    sessions = {}
    per_session = []
    used_fallback = tok_bin is None
    for path in sorted(session_files):
        name = os.path.basename(path)
        raw_turns = load_session_turns(path)
        if len(raw_turns) < min_turns:
            print(f"  skip {name}: {len(raw_turns)} turns < min {min_turns}")
            continue
        turns_ids = []
        ok = True
        for r in raw_turns:
            ids = tokenize_llama(r, model, tok_bin) if tok_bin else None
            if ids is None:
                ok = False
                break
            turns_ids.append(ids)
        if not ok:
            print(f"  skip {name}: tokenization failed")
            continue
        res = analyze_session(turns_ids, n_match, K)
        if res is None:
            continue
        res["session"] = name
        per_session.append(res)
        print(f"  {name}: turns={res['turns']} (W={res['warm_turns']}) "
              f"pc_served={res['prefix_cache_served_frac']*100:.0f}%  "
              f"τ_warm_full={res['tau_warm_full']:.2f} τ_cold_full={res['tau_cold_full']:.2f} "
              f"| τ_warm_suf={res['tau_warm_suffix']:.2f} τ_cold_suf={res['tau_cold_suffix']:.2f}")

    if not per_session:
        sys.exit("no sessions qualified (need >= min-turns and a tokenizer)")

    def pooled(key):
        s = sum(r["_pool"][key][0] for r in per_session)
        e = sum(r["_pool"][key][1] for r in per_session)
        return round(e / s, 3) if s else 0.0

    def spread(key):
        vals = sorted(r[key] for r in per_session)
        return {"min": vals[0], "median": round(median(vals), 3), "max": vals[-1]}

    pool = {k: pooled(k) for k in ("warm_full", "cold_full", "warm_suffix", "cold_suffix")}
    spr = {k: spread(f"tau_{k}") for k in ("warm_full", "cold_full", "warm_suffix", "cold_suffix")}

    wsuf = pool["warm_suffix"]
    csuf = pool["cold_suffix"]
    specialization = round(wsuf - csuf, 3)
    if wsuf >= GO_TAU and specialization >= SPECIALIZATION_MARGIN:
        verdict = "GO"
    elif wsuf >= MARGINAL_TAU and specialization >= SPECIALIZATION_MARGIN:
        verdict = "MARGINAL"
    else:
        verdict = "NO-GO"
    # diagnose WHY when not GO
    if verdict != "GO":
        if pool["warm_full"] >= GO_TAU and wsuf < GO_TAU:
            reason = ("warm τ lift is PREFIX-CACHE DOUBLE-COUNT: tau_warm_full "
                      f"{pool['warm_full']} clears the gate only because it counts the "
                      "leading shared prefix the default-on L1.2 cache already restores; "
                      f"on the post-prefix suffix tau_warm_suffix={wsuf} ~ "
                      f"tau_cold_suffix={csuf}. No ADDITIVE draft value.")
        elif specialization < SPECIALIZATION_MARGIN:
            reason = (f"warm history adds ~nothing over cold on the recomputed suffix "
                      f"(Δ={specialization}); in-prompt/in-region PLD already captures "
                      "the user's repeats (the design's honest prior).")
        else:
            reason = f"tau_warm_suffix={wsuf} below the {MARGINAL_TAU} bar."
    else:
        reason = (f"user-history warm-start lifts the recomputed-suffix draft to "
                  f"τ={wsuf} (+{specialization} over cold), additive to the shipped prefix cache.")

    out_obj = {
        "oracle": "spec_accept_draft_tuning_warmstart",
        "lever": "§8 L3.1 (b) draft tuning — per-user warm-started n-gram draft",
        "label": "ESTIMATE (git-history session proxy for a user; production number from usage_capture logs)",
        "tokenizer": "llama-tokenize" if tok_bin else "MISSING",
        "n_match": n_match, "K": K,
        "n_sessions_scored": len(per_session),
        "generic_baseline_tau": GENERIC_TAU,
        "thresholds": {"GO_tau": GO_TAU, "MARGINAL_tau": MARGINAL_TAU,
                       "specialization_margin": SPECIALIZATION_MARGIN,
                       "gate_tau": 2.5},
        "pooled_tau": pool,
        "spread_across_sessions": spr,
        "specialization_delta_suffix": specialization,
        "verdict": verdict,
        "reason": reason,
        "per_session": [{k: v for k, v in r.items() if k != "_pool"} for r in per_session],
        "note": (
            "tau_*_full counts ALL held-out tokens (incl. the leading shared prefix "
            "the SHIPPED default-on L1.2 prefix cache already restores -> upper proxy, "
            "NOT additive). tau_*_suffix counts ONLY the post-shared-prefix tokens the "
            "prefix cache does NOT serve -> the ADDITIVE draft value. warm=index seeded "
            "on the session's prior-half turns (user history); cold=empty seed. GO needs "
            "tau_warm_suffix>=1.8 AND (tau_warm_suffix-tau_cold_suffix)>=0.2. Draft is "
            "lossless (verifier emits) so any lift is free + zero regression risk."),
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(out_obj, open(out, "w"), indent=2)

    print()
    print(f"=== POOLED (n={n_match},K={K}, {len(per_session)} sessions) ===")
    print(f"  generic baseline τ            = {GENERIC_TAU}")
    print(f"  tau_warm_full   = {pool['warm_full']:.3f}  (UPPER proxy — incl. prefix-cache-served)")
    print(f"  tau_cold_full   = {pool['cold_full']:.3f}")
    print(f"  tau_warm_suffix = {pool['warm_suffix']:.3f}  (ADDITIVE — post-prefix-cache)   spread {spr['warm_suffix']}")
    print(f"  tau_cold_suffix = {pool['cold_suffix']:.3f}                                   spread {spr['cold_suffix']}")
    print(f"  specialization Δ(suffix) = {specialization:+.3f}")
    print(f"VERDICT = {verdict}  [{out}]")
    print(f"  {reason}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokens", nargs="?",
                    help="single-stream mode: llama-tokenize -f dump (id -> 'piece' per line)")
    ap.add_argument("--sessions", metavar="DIR",
                    help="L3.1 warm-start mode: dir of per-session .jsonl "
                         "(one ordered request record per line)")
    ap.add_argument("--model", default="models/qwen2.5-3b-instruct-q4_k_m.gguf",
                    help="gguf for llama-tokenize (sessions mode)")
    ap.add_argument("--tokenize-bin", default=os.environ.get("TOKENIZE_BIN", "llama-tokenize"))
    ap.add_argument("--n-match", type=int, default=2)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--min-turns", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.sessions:
        import glob
        import shutil
        files = glob.glob(os.path.join(args.sessions, "*.jsonl"))
        if not files:
            sys.exit(f"no .jsonl under {args.sessions}")
        tok_bin = args.tokenize_bin if shutil.which(args.tokenize_bin) else None
        if not tok_bin:
            sys.exit(f"llama-tokenize ('{args.tokenize_bin}') not on PATH; "
                     "sessions mode needs real BPE for fidelity")
        if not os.path.isfile(args.model):
            sys.exit(f"model {args.model} missing (needed by llama-tokenize)")
        out = args.out or "reports/oracle/spec_accept_warmstart.json"
        run_sessions(files, args.model, tok_bin, args.n_match, args.k,
                     args.min_turns, out)
        return

    if not args.tokens:
        ap.error("provide a positional tokens dump (single-stream) or --sessions DIR")
    ids = load_ids(args.tokens)
    if len(ids) < 100:
        sys.exit(f"too few tokens ({len(ids)})")
    run_single_stream(ids, args.out or "reports/oracle/spec_accept.json")


if __name__ == "__main__":
    main()
