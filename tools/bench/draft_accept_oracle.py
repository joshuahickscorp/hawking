#!/usr/bin/env python3
"""Small-DENSE-DRAFT speculative-decoding ACCEPTANCE oracle (axis-3 spec, the
reframe the EAGLE/EAGLE-3 trained-head kill does NOT touch).

WHAT THIS IS
------------
The offline decision tool for "use a small dense Qwen (0.5B or 1.5B Q4_K_M) to
draft for the 3B target". It does two separable jobs:

  (1) ACCEPT MATH on SAVED logits — given target-logits and draft-logits over a
      code corpus, compute the speculative-sampling acceptance distribution and
      the MEAN ACCEPTED LENGTH per cycle (`tau`), under the exact accept rule a
      real lossless spec-decode runtime uses (Leviathan/Chen/Google "Fast
      Inference from Transformers via Speculative Decoding", 2023; the same rule
      DeepMind's SpS uses). This part is CPU/NumPy and contamination-immune.

  (2) GO/NO-GO from `tau` + the per-cycle COST of running the draft. Unlike the
      n-gram oracle (`oracle_spec_accept.py`), where the draft is a ~free CPU
      automaton so `tau` IS the speedup ceiling, a DENSE draft costs real
      forward passes. So `tau` alone is NOT the verdict — it must clear a
      BREAKEVEN that depends on the draft/target forward-cost ratio `c`, the
      block size `k`, and the baseline we are beating (plain autoregressive, OR
      the user-ngram "bonus-first" path that already spends 2 target forwards
      per cycle). The inequality is in `speedup_vs_*` below and in the report.

WHY THIS IS GPU-FREE / WHAT IS THE COLAB-or-GPU-LANE STEP
--------------------------------------------------------
The accept math needs the two models' next-token logit streams on the SAME
code token sequence. Producing those logits is a model forward pass = the
GPU/Colab step. This file CANNOT and DOES NOT run the models. It:
  * `--selftest`         : synthetic agreeing/disagreeing logit streams with a
                           KNOWN accept length; asserts the computed tau, the
                           accept rule, and the breakeven algebra. Exits non-zero
                           on any failure. (in-session gate)
  * `--logits T.npy D.npy`: the REAL accept math, consuming logits the GPU lane
                           exported (target T, draft D, aligned, same tokens).
  * `--sweep`            : print the GO/NO-GO threshold table (tau breakeven as a
                           function of c, k) so the lane knows the target tau
                           BEFORE it spends GPU time — and `--c`/`--tau` plug a
                           measured pair straight into the verdict.

The decisive gate is therefore: run BOTH models on held-out CODE, dump aligned
logits, feed them here; AND measure the real paired draft/target forward-cost
ratio `c` on-device (it is NOT exactly the byte ratio — see the report). This
oracle supplies the contract + the threshold so that GPU spend is decisive, not
exploratory ("a wrong simulation is worse than none" — bible §8.3.1).

COST-RATIO GROUNDING (label = ESTIMATE; the on-device paired bench is decisive)
------------------------------------------------------------------------------
Decode on this engine is bandwidth-bound (~85% GPU-busy; Q4_K predec GEMV at the
HW memory-model optimum — MEMORY.md). A decode step streams the whole model once,
so the per-token forward cost ratio ~ the total-tensor-byte ratio. Measured from
the local GGUFs (header byte counts, no dequant):
      0.5B / 3B tensor-bytes = 0.252      1.5B / 3B = 0.578
So c_0.5B ~ 0.25, c_1.5B ~ 0.58 (ESTIMATE). True c is likely a touch LOWER for
the draft (KV cache + per-dispatch overhead scale sub-linearly with model size,
favoring the smaller model), and a touch HIGHER if draft GEMVs underfill the GPU.
Carry [0.22, 0.30] for 0.5B and [0.52, 0.65] for 1.5B until the paired bench pins
it. `RATIO_DEFAULTS` below holds these; `--c` overrides with a measured value.

RAM discipline: logits are loaded one array pair at a time; no model load here.
RSS ceiling 3 GB (the .npy logit dumps for a few-k-token corpus are << 1 GB).
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
REPORT_JSON = str(ROOT / "reports/oracle/small_draft_accept.json")
SEED = 0

# Byte-ratio-grounded forward-cost ESTIMATES (draft cost in units of one 3B
# forward = 1.0). Decode is bandwidth-bound, so cost ~ tensor bytes streamed.
# point = local GGUF tensor-byte ratio; lo/hi bracket the sub-/super-linear
# corrections (KV + dispatch overhead vs GPU underfill). The on-device paired
# forward-cost bench is the decisive measurement that collapses each interval.
RATIO_DEFAULTS = {
    "0.5B": {"point": 0.25, "lo": 0.22, "hi": 0.30, "tensor_byte_ratio": 0.252},
    "1.5B": {"point": 0.58, "lo": 0.52, "hi": 0.65, "tensor_byte_ratio": 0.578},
}
# Baseline the dense draft must beat. Plain autoregressive = 1.0 tok / target
# forward. The user-ngram "bonus-first" path (task premise) spends 2 target
# forwards per cycle and emits tau_ngram tokens; generic code tau_ngram = 1.43
# (reports/oracle/spec_accept.json), so its rate = 1.43/2 = 0.715 tok/forward —
# i.e. the ngram bonus-first path is actually SLOWER than plain AR on generic
# code. The honest baseline a dense draft must beat is therefore max(1.0,
# tau_ngram/2). We report the threshold against BOTH so the lane sees which bites.
TAU_NGRAM_GENERIC = 1.43

try:
    import resource

    def rss_gb():
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / (1024**3) if sys.platform == "darwin" else kb / (1024**2)
except Exception:  # pragma: no cover
    def rss_gb():
        return float("nan")


# ============================================================================
# Speculative-sampling accept rule (Leviathan/Chen, lossless).
# For each drafted position i the draft proposed token x_i ~ q (draft dist).
# The verifier accepts x_i with prob min(1, p(x_i)/q(x_i)) where p is the TARGET
# dist; on the FIRST reject it stops and the target emits one correction token
# (the "+1" bonus) sampled from the residual (p-q)_+; if ALL k are accepted the
# target emits one bonus token from its own dist. So tokens emitted in a cycle =
# (#accepted) + 1, with #accepted in [0, k]. tau = E[tokens emitted per cycle].
#
# At TEMPERATURE 0 (greedy, the dismantle default decode) this degenerates to
# EXACT-MATCH acceptance: accept x_i iff x_i == argmax p_i, i.e. iff the draft's
# greedy token equals the target's greedy token. We expose both regimes because
# the production decode is greedy but a sampling server is a real deployment too.
# ============================================================================
def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def accept_lengths_greedy(target_logits, draft_logits):
    """tau under temp=0 greedy verification (the dismantle default).

    target_logits, draft_logits : (T, V) aligned next-token logits over ONE
    contiguous token stream (position t predicts token t+1). We slice the stream
    into consecutive cycles of up to k drafted positions; within a cycle the
    draft's greedy token is accepted iff it equals the target's greedy token,
    stopping at the first mismatch. This is the deterministic special case of
    the Leviathan rule and is what a greedy lossless runtime actually does.

    Returns a callable taking k -> dict (so we factor the per-position
    agreement once and reslice cheaply for every k).
    """
    tgt = np.asarray(target_logits)
    drf = np.asarray(draft_logits)
    assert tgt.shape == drf.shape and tgt.ndim == 2, "logits must be aligned (T,V)"
    agree = (np.argmax(tgt, -1) == np.argmax(drf, -1))  # (T,) per-position greedy match
    return _Acc(agree)


def accept_lengths_sampled(target_logits, draft_logits, temperature=1.0,
                           rng=None):
    """tau under the full stochastic Leviathan rule at a given temperature.

    The draft token at each position is itself sampled from q (the draft dist);
    acceptance is Bernoulli(min(1, p/q)). We Monte-Carlo a single draft token
    per position (the runtime drafts once) and compute the accept indicator.
    Returns the same _Acc helper. EXPECTATION note: for an unbiased per-position
    accept PROBABILITY one can also use the closed form E_q[min(1,p/q)] =
    sum_x min(p_x, q_x) (total-variation overlap); `--exact-accept-prob` uses
    that instead of a sample. The sliced tau then uses these per-position
    accept probabilities as independent (a standard, documented approximation —
    real runs have mild within-cycle correlation; the GPU lane's measured tau on
    real logits is the ground truth this approximates).
    """
    tgt = np.asarray(target_logits, dtype=np.float64)
    drf = np.asarray(draft_logits, dtype=np.float64)
    assert tgt.shape == drf.shape and tgt.ndim == 2
    if rng is None:
        rng = np.random.default_rng(SEED)
    p = _softmax(tgt / temperature)
    q = _softmax(drf / temperature)
    # draft samples one token per position from q
    cum = np.cumsum(q, axis=-1)
    u = rng.random((q.shape[0], 1))
    x = (u < cum).argmax(axis=-1)               # sampled draft token per position
    rows = np.arange(q.shape[0])
    ratio = np.minimum(1.0, p[rows, x] / np.maximum(q[rows, x], 1e-30))
    accept = rng.random(q.shape[0]) < ratio     # Bernoulli accept indicator
    return _Acc(accept)


def accept_prob_overlap(target_logits, draft_logits, temperature=1.0):
    """Per-position EXPECTED accept probability = sum_x min(p_x,q_x) (the
    expectation of the Leviathan accept indicator over the draft's own sampling).
    Deterministic (no RNG). Returns _AccProb (continuous accept probs)."""
    p = _softmax(np.asarray(target_logits, np.float64) / temperature)
    q = _softmax(np.asarray(draft_logits, np.float64) / temperature)
    a = np.minimum(p, q).sum(axis=-1)           # (T,) accept prob per position
    return _AccProb(a)


class _Acc:
    """Holds a per-position boolean accept array; slices it into cycles of size
    k and computes the acceptance histogram + tau. EVERY cycle emits accepted+1
    tokens (the target's correction/bonus token), exactly as a lossless runtime."""

    def __init__(self, accept_bool):
        self.a = np.asarray(accept_bool, dtype=bool)

    def tau(self, k):
        N = self.a.size
        i = 0
        hist = {}
        steps = emitted = 0
        while i < N:
            j = 0
            while j < k and i + j < N and self.a[i + j]:
                j += 1
            # j accepted; cycle emits j+1 (bonus token), advances i by j+1
            hist[j] = hist.get(j, 0) + 1
            steps += 1
            emitted += j + 1
            i += j + 1
        tau = emitted / steps if steps else 0.0
        return {
            "k": k, "steps": steps, "emitted": emitted,
            "mean_accepted_len": round(tau, 4),
            "per_position_accept_rate": round(float(self.a.mean()), 4),
            "accept_hist": {str(a): hist[a] for a in sorted(hist)},
        }


class _AccProb:
    """Per-position accept PROBABILITY array -> EXPECTED tau for block k, under
    the independence approximation. With per-position accept prob a_i, the
    expected #accepted before first reject in a cycle starting at position s is
    sum_{m>=1} prod_{i<m} a_{s+i} (capped at k); emitted = that + 1. We compute it
    in closed form per cycle. This is the stochastic-temperature analogue of the
    greedy exact count, and is deterministic given the logits."""

    def __init__(self, accept_prob):
        self.a = np.clip(np.asarray(accept_prob, dtype=np.float64), 0.0, 1.0)

    def tau(self, k):
        N = self.a.size
        i = 0
        steps = 0
        emitted = 0.0
        accepted_total = 0.0
        while i < N:
            # expected accepted in this cycle = sum_{m=1..k} prod_{t=0..m-1} a[i+t]
            prod = 1.0
            exp_acc = 0.0
            for m in range(k):
                if i + m >= N:
                    break
                prod *= self.a[i + m]
                exp_acc += prod
            emitted += exp_acc + 1.0          # +1 bonus token
            accepted_total += exp_acc
            steps += 1
            # advance by the EXPECTED emitted, rounded to keep cycles ~ disjoint;
            # for tau the ratio emitted/steps is what matters, and steps counts
            # one verify forward per cycle. Advance by ceil(exp_acc)+1 so cycles
            # tile the stream without overlap (a conservative, documented tiling).
            i += int(np.ceil(exp_acc)) + 1
        tau = emitted / steps if steps else 0.0
        return {
            "k": k, "steps": steps, "emitted": round(emitted, 2),
            "mean_accepted_len": round(tau, 4),
            "mean_per_position_accept_prob": round(float(self.a.mean()), 4),
            "accept_hist": None,   # continuous; no integer histogram
        }


# ============================================================================
# Breakeven / speedup algebra (the GO/NO-GO core).
# Per CYCLE: draft proposes k tokens SEQUENTIALLY (k draft forwards, cost k*c),
# then ONE target forward (cost 1.0 + verify overhead v, where v folds the
# logit-compare + KV bookkeeping; ~0 on a fused runtime) verifies all k in
# parallel and emits the bonus token. yield = tau tokens.
#   spec rate (tokens / target-forward-equivalent time) = tau / (k*c + 1 + v)
# Baselines:
#   plain AR rate          = 1.0
#   ngram bonus-first rate = tau_ngram / 2     (2 target forwards/cycle, ~free draft)
# Speedups:
#   S_ar  = tau / (k*c + 1 + v)
#   S_ng  = [tau / (k*c + 1 + v)] / (tau_ngram / 2)
# GO (beats plain AR)        <=> tau >  (k*c + 1 + v)
# GO (beats ngram b-first)   <=> tau >  (tau_ngram/2)*(k*c + 1 + v)
# ============================================================================
def speedup_vs_ar(tau, c, k, v=0.0):
    return tau / (k * c + 1.0 + v)


def speedup_vs_ngram(tau, c, k, tau_ngram=TAU_NGRAM_GENERIC, v=0.0):
    base = tau_ngram / 2.0
    return speedup_vs_ar(tau, c, k, v) / base if base > 0 else float("inf")


def tau_threshold_ar(c, k, v=0.0):
    """Min tau to merely break even vs plain autoregressive."""
    return k * c + 1.0 + v


def tau_threshold_ngram(c, k, tau_ngram=TAU_NGRAM_GENERIC, v=0.0):
    """Min tau to beat the ngram bonus-first path."""
    return (tau_ngram / 2.0) * (k * c + 1.0 + v)


def best_k_for_tau_curve(c, tau_of_k, k_grid=(2, 3, 4, 5, 6, 8), v=0.0):
    """Given a tau(k) function, pick the k maximizing speedup vs AR."""
    best = None
    for k in k_grid:
        t = tau_of_k(k)
        s = speedup_vs_ar(t, c, k, v)
        if best is None or s > best[1]:
            best = (k, s, t)
    return {"best_k": best[0], "speedup_vs_ar": round(best[1], 4),
            "tau_at_best_k": round(best[2], 4)}


def tau_from_alpha(alpha, k):
    """Expected tokens emitted per cycle (incl. the +1 bonus) for a geometric
    accept model with constant per-token acceptance `alpha`, block k. Used ONLY
    in --sweep to translate a literature/measured per-token accept rate into a
    tau, NOT in the real --logits path (which measures tau directly)."""
    if abs(alpha - 1.0) < 1e-12:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


# ============================================================================
# --selftest : synthetic streams with KNOWN accept length; assert the math.
# ============================================================================
def selftest():
    rng = np.random.default_rng(SEED)
    fails = []
    V = 64

    # 1) FULLY AGREEING greedy stream: draft argmax == target argmax everywhere.
    #    Then within a block of k, all k accept => emitted k+1 per cycle => tau=k+1.
    #    Use T = exact multiple of (k+1) per k so the last cycle is NOT truncated
    #    (a finite stream tiled in (k+1)-steps only hits k+1 exactly when (k+1)|T;
    #    the truncated-tail case is a real boundary effect, not a bug).
    def agreeing_tau(k):
        T = (k + 1) * 120
        base = rng.standard_normal((T, V))
        col = np.arange(T) % V
        tgt = base.copy(); tgt[np.arange(T), col] += 10.0
        drf = base.copy(); drf[np.arange(T), col] += 8.0   # same argmax, diff logits
        return accept_lengths_greedy(tgt, drf).tau(k)["mean_accepted_len"]

    agree_tau = {}
    for k in (4, 8):
        agree_tau[k] = agreeing_tau(k)
        if abs(agree_tau[k] - (k + 1)) > 1e-9:
            fails.append(f"agreeing stream k={k}: tau={agree_tau[k]} != {k+1}")

    # 2) FULLY DISAGREEING greedy stream: draft argmax never == target argmax.
    #    Every cycle: 0 accepted => emitted 1 => tau=1.0 for any k.
    T = 600
    base = rng.standard_normal((T, V))
    tgt2 = base.copy(); tgt2[np.arange(T), 0] += 10.0     # argmax = col 0
    drf2 = base.copy(); drf2[np.arange(T), 1] += 10.0     # argmax = col 1
    acc2 = accept_lengths_greedy(tgt2, drf2)
    for k in (4, 8):
        r = acc2.tau(k)
        if abs(r["mean_accepted_len"] - 1.0) > 1e-9:
            fails.append(f"disagreeing stream k={k}: tau={r['mean_accepted_len']} != 1.0")

    # 3) KNOWN PATTERN: accept exactly 2 then a miss, repeating (a=[T,T,F,...]).
    #    With k>=2 every cycle accepts 2, emits 3, advances 3 => tau=3.0 exactly.
    patt = np.array(([True, True, False] * 400)[:1200])
    accp = _Acc(patt)
    for k in (3, 4, 8):
        r = accp.tau(k)
        if abs(r["mean_accepted_len"] - 3.0) > 1e-9:
            fails.append(f"TTF pattern k={k}: tau={r['mean_accepted_len']} != 3.0")
    # with k=1 (draft only 1 token): each cycle accepts<=1; pattern T,T,F,T,T,F...
    # cycle1 i=0 accept a[0]=T ->1 accepted, emit2, i=2; a[2]=F emit1 i=3; a[3]=T emit2 i=5...
    r1 = accp.tau(1)
    # not asserting a closed form for k=1; just that it is in (1, 2]
    if not (1.0 < r1["mean_accepted_len"] <= 2.0):
        fails.append(f"TTF k=1 tau out of range: {r1['mean_accepted_len']}")

    # 4) accept-prob overlap: identical dists -> overlap 1.0 -> accept prob 1 ->
    #    expected tau = k+1 (use T a multiple of k+1 so no truncated tail);
    #    orthogonal dists -> overlap 0 -> tau = 1.0.
    Lov = rng.standard_normal((6 * 100, V))               # (k+1)=6 divides T
    same = accept_prob_overlap(Lov, Lov)                  # p==q overlap=1
    if abs(same.tau(5)["mean_accepted_len"] - 6.0) > 1e-6:
        fails.append(f"overlap identical tau != 6 (got {same.tau(5)['mean_accepted_len']})")
    onehotA = np.full((50, V), -30.0); onehotA[:, 0] = 30.0
    onehotB = np.full((50, V), -30.0); onehotB[:, 1] = 30.0
    disj = accept_prob_overlap(onehotA, onehotB)
    if abs(disj.tau(5)["mean_accepted_len"] - 1.0) > 1e-3:
        fails.append(f"overlap disjoint tau != 1 (got {disj.tau(5)['mean_accepted_len']})")

    # 5) breakeven algebra: at tau == threshold, speedup == 1 (vs AR) / matches base.
    for c in (0.25, 0.58):
        for k in (4, 8):
            t_ar = tau_threshold_ar(c, k)
            if abs(speedup_vs_ar(t_ar, c, k) - 1.0) > 1e-9:
                fails.append(f"AR breakeven not unity c={c} k={k}")
            t_ng = tau_threshold_ngram(c, k)
            if abs(speedup_vs_ngram(t_ng, c, k) - 1.0) > 1e-9:
                fails.append(f"ngram breakeven not unity c={c} k={k}")
    # monotonicity: higher c => higher tau threshold (more expensive draft)
    if not (tau_threshold_ar(0.58, 4) > tau_threshold_ar(0.25, 4)):
        fails.append("threshold not increasing in c")

    # 6) tau_from_alpha sanity: alpha->1 gives k+1; alpha=0 gives 1.0
    if abs(tau_from_alpha(1.0, 5) - 6.0) > 1e-9 or abs(tau_from_alpha(0.0, 5) - 1.0) > 1e-9:
        fails.append("tau_from_alpha endpoints wrong")
    if not (tau_from_alpha(0.7, 8) > tau_from_alpha(0.7, 4)):
        fails.append("tau_from_alpha not increasing in k")

    print("=== small-dense-draft accept oracle self-test ===")
    print(f"  agreeing greedy stream tau == k+1 .......... OK (k=4->{agree_tau[4]}, k=8->{agree_tau[8]})")
    print(f"  disagreeing greedy stream tau == 1.0 ....... OK")
    print(f"  TTF pattern tau == 3.0 (k>=2) .............. OK")
    print(f"  accept-prob overlap endpoints (1.0 / k+1) .. OK")
    print(f"  breakeven algebra unity at threshold ....... OK")
    print(f"  tau(alpha,k) endpoints + monotone .......... OK")
    if fails:
        print("\nSELF-TEST FAILED:")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("\nself-test PASSED — accept math + breakeven algebra are trustworthy.")
    return True


# ============================================================================
# --logits : REAL accept math on GPU-lane-exported aligned logit streams.
# ============================================================================
def run_logits(target_npy, draft_npy, temperature, k_grid, c_point, draft_tag,
               exact_prob, out):
    t0 = time.time()
    tgt = np.load(target_npy)
    drf = np.load(draft_npy)
    if tgt.shape != drf.shape or tgt.ndim != 2:
        sys.exit(f"[FATAL] logits must be aligned 2-D (T,V); got {tgt.shape} vs {drf.shape}")
    g = rss_gb()
    if g > 3.0:
        sys.exit(f"[FATAL] RSS {g:.2f} GB > 3 GB after loading logits")

    greedy = accept_lengths_greedy(tgt, drf)
    grows = [greedy.tau(k) for k in k_grid]
    sampled = None
    if temperature and temperature > 0:
        srows = (accept_prob_overlap(tgt, drf, temperature) if exact_prob
                 else accept_lengths_sampled(tgt, drf, temperature))
        sampled = [srows.tau(k) for k in k_grid]

    # verdict at each k for the chosen draft cost
    def verdict_rows(rows):
        out_rows = []
        for r in rows:
            k = r["k"]; tau = r["mean_accepted_len"]
            s_ar = speedup_vs_ar(tau, c_point, k)
            s_ng = speedup_vs_ngram(tau, c_point, k)
            out_rows.append({
                **r,
                "c_used": c_point,
                "speedup_vs_plain_ar": round(s_ar, 4),
                "speedup_vs_ngram_bonus_first": round(s_ng, 4),
                "tau_threshold_ar": round(tau_threshold_ar(c_point, k), 4),
                "tau_threshold_ngram": round(tau_threshold_ngram(c_point, k), 4),
                "beats_ar": bool(s_ar > 1.0),
            })
        return out_rows

    gv = verdict_rows(grows)
    sv = verdict_rows(sampled) if sampled else None
    best_ar = max(gv, key=lambda r: r["speedup_vs_plain_ar"])
    verdict = "GO" if best_ar["speedup_vs_plain_ar"] > 1.0 else "NO-GO"

    obj = {
        "oracle": "small_dense_draft_accept",
        "label": "MEASURED accept math on GPU-lane logits (cost ratio c is ESTIMATE unless --c from paired bench)",
        "draft_tag": draft_tag,
        "temperature": temperature,
        "tokens": int(tgt.shape[0]),
        "vocab": int(tgt.shape[1]),
        "c_used": c_point,
        "tau_ngram_baseline": TAU_NGRAM_GENERIC,
        "greedy": gv,
        "sampled": sv,
        "best_greedy_vs_ar": {k: best_ar[k] for k in
                              ("k", "mean_accepted_len", "speedup_vs_plain_ar",
                               "speedup_vs_ngram_bonus_first")},
        "verdict_greedy_vs_ar": verdict,
        "note": ("tau = tokens emitted per verify forward under the lossless "
                 "Leviathan accept rule (greedy = exact argmax match; sampled = "
                 "TV-overlap accept prob). Verdict needs tau > k*c+1 to beat plain "
                 "AR. c is the draft/target forward-cost ratio; pass --c from the "
                 "on-device paired bench to make the verdict decisive."),
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(obj, open(out, "w"), indent=2)
    print(f"tokens={tgt.shape[0]} vocab={tgt.shape[1]} draft={draft_tag} c={c_point} temp={temperature}")
    print("  [greedy / temp=0]")
    for r in gv:
        flag = "GO" if r["beats_ar"] else "  "
        print(f"   k={r['k']:2d}: tau={r['mean_accepted_len']:.3f}  "
              f"S_ar={r['speedup_vs_plain_ar']:.3f} {flag}  "
              f"(need tau>{r['tau_threshold_ar']:.2f})  "
              f"S_ngram={r['speedup_vs_ngram_bonus_first']:.3f}")
    if sv:
        print(f"  [sampled / temp={temperature}]")
        for r in sv:
            print(f"   k={r['k']:2d}: tau={r['mean_accepted_len']:.3f}  "
                  f"S_ar={r['speedup_vs_plain_ar']:.3f}")
    print(f"VERDICT (greedy vs plain AR) = {verdict}  best k={best_ar['k']} "
          f"S_ar={best_ar['speedup_vs_plain_ar']:.3f}  [{out}]")
    print(f"wall={time.time()-t0:.1f}s RSS={rss_gb():.2f} GB")
    return obj


# ============================================================================
# --sweep : print the GO/NO-GO threshold table (no logits needed).
# ============================================================================
def run_sweep(k_grid, alphas):
    print("=== tau BREAKEVEN thresholds (min mean-accepted-length to win) ===")
    print(f"  baselines: plain-AR rate=1.0 ; ngram-bonus-first rate=tau_ngram/2 "
          f"(tau_ngram={TAU_NGRAM_GENERIC} -> {TAU_NGRAM_GENERIC/2:.3f})\n")
    for tag, d in RATIO_DEFAULTS.items():
        print(f"-- {tag} draft, c={d['point']} (byte-ratio {d['tensor_byte_ratio']:.3f}; "
              f"bracket [{d['lo']},{d['hi']}]) --")
        print(f"   {'k':>3} | {'tau>X beat AR':>14} | {'tau>X beat ngram':>16}")
        for k in k_grid:
            print(f"   {k:>3} | {tau_threshold_ar(d['point'], k):>14.2f} | "
                  f"{tau_threshold_ngram(d['point'], k):>16.2f}")
        print()
    print("=== achievable tau from a per-token accept rate alpha (geometric model) ===")
    print("   (use to read whether a plausible alpha clears the thresholds above)\n")
    hdr = "   alpha | " + " | ".join(f"k={k:<2d}" for k in k_grid)
    print(hdr)
    for a in alphas:
        print(f"   {a:>5.2f} | " + " | ".join(f"{tau_from_alpha(a,k):>4.2f}" for k in k_grid))
    print()
    print("READ: cross the two tables. e.g. 0.5B (c=0.25), k=4 needs tau>2.00 to beat "
          "AR; an alpha~0.6 draft gives tau~2.31 -> clears. 1.5B (c=0.58), k=4 needs "
          "tau>3.32; needs alpha~0.8. The GPU lane measures the REAL tau via --logits.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="validate accept math + breakeven on synthetic (in-session gate)")
    ap.add_argument("--logits", nargs=2, metavar=("TARGET_NPY", "DRAFT_NPY"),
                    help="REAL accept math on aligned (T,V) logit dumps from the GPU lane")
    ap.add_argument("--sweep", action="store_true",
                    help="print GO/NO-GO tau threshold tables (no logits needed)")
    ap.add_argument("--draft", choices=list(RATIO_DEFAULTS), default="0.5B",
                    help="which draft's cost-ratio default to use for the verdict")
    ap.add_argument("--c", type=float, default=None,
                    help="MEASURED draft/target forward-cost ratio (overrides byte-ratio default)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy exact-match (dismantle default); >0 adds the sampled regime")
    ap.add_argument("--exact-accept-prob", action="store_true",
                    help="with temp>0, use TV-overlap accept prob (deterministic) not a sample")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8],
                    help="block sizes to evaluate")
    ap.add_argument("--out", default=REPORT_JSON)
    args = ap.parse_args()

    if not (args.selftest or args.logits or args.sweep):
        ap.error("pick one of --selftest / --logits T D / --sweep")
    if args.selftest:
        selftest()
    if args.sweep:
        run_sweep(tuple(args.k), (0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    if args.logits:
        c = args.c if args.c is not None else RATIO_DEFAULTS[args.draft]["point"]
        run_logits(args.logits[0], args.logits[1], args.temperature, args.k,
                   c, args.draft, args.exact_accept_prob, args.out)


if __name__ == "__main__":
    main()
