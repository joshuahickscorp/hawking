#!/usr/bin/env python3.12
"""REAL routing-frequency calibration for Qwen3-235B-A22B.

Replaces the deterministic expert-index stand-in used by the routing-aware allocation lever
(whole-model 0.4930 -> 0.3554) with measured router statistics from the true resident source.

What it does: runs the real qwen_real_forward residual stream over the 6-prompt campaign holdout
(the SAME prompts qwen_correction_wave scores on) and, at every one of the 94 layers, records for
all 128 experts:
  * top8_count      how often the expert landed in the top-8
  * topk_mass       summed renormalized router weight it actually received
  * softmax_mass    summed full-softmax probability over ALL experts (selected or not)

Two structural shortcuts vs the campaign forward, both exact for routing:
  * lm_head and the final norm are skipped (routing never reads them).
  * MoE is restructured expert-OUTER: router logits for every holdout token are computed first,
    then each distinct expert is streamed ONCE per layer and applied to all of its assigned tokens
    across all 6 prompts at once. Same arithmetic, ~6x fewer shard reads than per-prompt streaming.

HONESTY. 88 holdout tokens x 8 picks = 704 router decisions spread over 128 experts is a SMALL
sample: the expected count per expert is 5.5. The sealed report quantifies that directly
(never-routed fraction, bootstrap stability of the hot/cold median split, and the binomial token
count that WOULD be needed). Read `trust_verdict` in the report before allocating bits on this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from qwen_correction_wave import HOLDOUT  # noqa: E402  same 6 prompts the campaign scores on
from qwen_real_forward import DEFAULT_META, from_source, rmsnorm, swiglu  # noqa: E402

REPORT = Path("reports/condense/general_frontier/QWEN3_235B_ROUTING_FREQUENCY.json")
SCHEMA = "hawking.qwen3_235b.routing_frequency.v1"
BOOTSTRAP = 2000
Z = 1.959963985  # two-sided 95 percent normal quantile


# ── the calibration pass ─────────────────────────────────────────────────────────────────────
def collect(max_layers: int | None = None, progress: bool = True,
            calib_tokens: int | None = None) -> dict:
    """Run the real forward and return raw per-layer routing selections.

    `calib_tokens` switches the source of the statistics from the SCORED holdout to the frozen,
    disjoint calibration corpus at the requested size. The sealed 88-token run fitted on the same
    six prompts it reported quality against - contamination - and its own analysis puts the stable
    median partition at roughly 979 tokens. Pass >= 1000 to satisfy both.
    """
    from tokenizers import Tokenizer  # type: ignore

    fwd = from_source()
    if not fwd.source_present():
        raise SystemExit("Qwen3-235B shards not resident; routing calibration needs the real source")
    g, r = fwd.g, fwd.reader
    tk = Tokenizer.from_file(str(DEFAULT_META / "tokenizer.json"))

    if calib_tokens:
        import qwen_calibration_corpus as CC  # noqa: PLC0415  optional path
        corpus = CC.build(min_tokens=calib_tokens, tokenizer=tk)
        prompts = [{"id": p["id"], "domain": p["domain"], "ids": p["ids"]}
                   for p in corpus["prompts"]]
        calib_source = {"kind": "frozen_disjoint_calibration_corpus",
                        "sha256": corpus["sha256"], "n_prompts": corpus["n_prompts"]}
    else:
        prompts = [{"id": h["id"], "domain": h["domain"], "ids": tk.encode(h["text"]).ids}
                   for h in HOLDOUT]
        calib_source = {"kind": "scored_holdout_CONTAMINATED", "sha256": None,
                        "n_prompts": len(prompts)}
    lens = [len(p["ids"]) for p in prompts]
    n_tok = sum(lens)
    xs = [r.bf16_rows("model.embed_tokens.weight", list(p["ids"])) for p in prompts]

    nb = g.n_layers if max_layers is None else min(max_layers, g.n_layers)
    sel_idx = np.zeros((nb, n_tok, g.top_k), dtype=np.int16)     # chosen experts per token
    sel_w = np.zeros((nb, n_tok, g.top_k), dtype=np.float32)     # renormalized top-k weights
    soft_mass = np.zeros((nb, g.n_experts), dtype=np.float64)    # full-softmax mass over all experts

    t_start = time.time()
    for L in range(nb):
        t0 = time.time()
        for i in range(len(prompts)):
            xs[i] = xs[i] + fwd._attention(L, xs[i])
        ln = r.bf16(f"model.layers.{L}.post_attention_layernorm.weight")
        h = np.concatenate([rmsnorm(x, ln, g.eps) for x in xs], axis=0)      # [n_tok, hidden]

        logits = h @ r.bf16(f"model.layers.{L}.mlp.gate.weight").T           # [n_tok, n_experts]
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        soft_mass[L] = probs.sum(axis=0)
        idx = np.argsort(-probs, axis=1)[:, :g.top_k]                        # [n_tok, top_k]
        w = np.take_along_axis(probs, idx, axis=1).astype(np.float32)
        if g.norm_topk_prob:
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-20)
        sel_idx[L], sel_w[L] = idx, w

        out = np.zeros_like(h)
        for e in np.unique(idx):                                             # expert OUTER
            rows, slot = np.nonzero(idx == e)
            ex = fwd._load_expert(L, int(e))
            hr = h[rows]
            a = swiglu(hr @ ex["gate"].T, hr @ ex["up"].T)
            out[rows] += w[rows, slot][:, None] * (a @ ex["down"].T)
        off = 0
        for i, n in enumerate(lens):
            xs[i] = xs[i] + out[off:off + n]
            off += n
        if progress:
            print(f"  layer {L:2d}/{nb-1}  {time.time()-t0:5.1f}s  experts={len(np.unique(idx)):3d}  "
                  f"elapsed={(time.time()-t_start)/60:5.1f}m", flush=True)

    return {"prompts": [{k: v for k, v in p.items() if k != "ids"} | {"n_tokens": len(p["ids"])}
                        for p in prompts],
            "n_tokens": n_tok, "n_layers": nb, "n_experts": g.n_experts, "top_k": g.top_k,
            "sel_idx": sel_idx, "sel_w": sel_w, "soft_mass": soft_mass,
            "calibration_source": calib_source,
            "wall_seconds": round(time.time() - t_start, 1)}


# ── statistics ───────────────────────────────────────────────────────────────────────────────
def _counts(sel_idx: np.ndarray, n_experts: int) -> np.ndarray:
    """[n_layers, n_tok, k] expert ids -> [n_layers, n_experts] top-8 counts."""
    return np.stack([np.bincount(a.ravel(), minlength=n_experts) for a in sel_idx]).astype(np.int64)


def _hot(counts_layer: np.ndarray, tie_break: np.ndarray) -> np.ndarray:
    """Boolean hot mask: top half by count, ties broken by softmax mass. Exactly half are hot."""
    order = np.lexsort((-tie_break, -counts_layer))
    hot = np.zeros(counts_layer.shape[0], dtype=bool)
    hot[order[: counts_layer.shape[0] // 2]] = True
    return hot


def count_vs_mass_agreement(counts: np.ndarray, soft: np.ndarray) -> float:
    """Fraction of (layer, expert) cells where the count-based hot/cold split agrees with the split
    from full-softmax mass. softmax_mass is the SAME underlying preference measured with far lower
    variance (every token contributes to every expert, no zeros), so disagreement here is a direct
    read on how much of the count-based partition is sampling noise."""
    agree = [ _hot(counts[L], soft[L]) == _hot(np.round(soft[L] * 1e6).astype(np.int64), soft[L])
              for L in range(counts.shape[0]) ]
    return round(float(np.mean(agree)), 4)


def tokens_needed(counts: np.ndarray, n_tok: int) -> dict:
    """Binomial token count that WOULD separate each expert from its layer median at 95 percent.

    count_e ~ Binomial(n_tok, p_e). The variance is floored at the NULL variance p_med(1-p_med):
    without that floor a never-routed expert (p_hat == 0) gets ~zero variance and a nonsensically
    small requirement, when it is in fact the least-determined expert in the layer.
    """
    p = counts / float(n_tok)
    p_med = np.median(p, axis=1, keepdims=True)
    margin = np.abs(p - p_med)
    var = np.maximum(p * (1.0 - p), p_med * (1.0 - p_med))
    with np.errstate(divide="ignore", invalid="ignore"):
        n_req = np.where(margin > 0, (Z * Z) * var / np.maximum(margin, 1e-12) ** 2, np.inf)
    finite = n_req[np.isfinite(n_req)]
    return {
        "tokens_needed_median_expert": (int(np.median(finite)) if finite.size else None),
        "tokens_needed_for_90pct_of_experts": (int(np.percentile(finite, 90))
                                               if finite.size else None),
        "experts_tied_with_layer_median_fraction": round(float((~np.isfinite(n_req)).mean()), 4),
        "binomial_note": "count_e ~ Binomial(n_tokens, p_e), variance floored at the null "
                         "p_med(1-p_med) so never-routed experts are not scored as easy. The 8 "
                         "picks per token are drawn without replacement and are correlated across "
                         "experts, so treat this as an order-of-magnitude floor, not a tight "
                         "requirement.",
    }


def analyse(raw: dict, bootstrap: int = BOOTSTRAP, seed: int = 20260720) -> dict:
    sel_idx, sel_w, soft = raw["sel_idx"], raw["sel_w"], raw["soft_mass"]
    nb, n_tok, k = sel_idx.shape
    n_e = raw["n_experts"]
    counts = _counts(sel_idx, n_e)

    topk_mass = np.zeros((nb, n_e), dtype=np.float64)
    for L in range(nb):
        np.add.at(topk_mass[L], sel_idx[L].ravel(), sel_w[L].ravel().astype(np.float64))

    hot = np.stack([_hot(counts[L], soft[L]) for L in range(nb)])            # [nb, n_e]

    # -- bootstrap: resample the 88 holdout TOKENS with replacement, re-derive the split ------
    rng = np.random.default_rng(seed)
    agree = np.zeros((nb, n_e), dtype=np.float64)
    for _ in range(bootstrap):
        pick = rng.integers(0, n_tok, n_tok)
        for L in range(nb):
            c = np.bincount(sel_idx[L][pick].ravel(), minlength=n_e).astype(np.int64)
            agree[L] += _hot(c, soft[L]) == hot[L]
    agree /= bootstrap

    never = counts == 0
    return {
        "counts": counts, "topk_mass": topk_mass, "soft_mass": soft, "hot": hot,
        "agree": agree,
        "sampling": {
            "n_tokens": int(n_tok), "router_decisions": int(n_tok * k),
            "expected_count_per_expert": round(n_tok * k / n_e, 3),
            "never_routed_pairs": int(never.sum()),
            "never_routed_fraction": round(float(never.mean()), 6),
            "layers_with_a_never_routed_expert": int((never.any(axis=1)).sum()),
            "max_never_routed_in_one_layer": int(never.sum(axis=1).max()),
            "bootstrap_resamples": int(bootstrap),
            "count_split_vs_softmax_mass_split_agreement": count_vs_mass_agreement(counts, soft),
            "hot_cold_stability_mean": round(float(agree.mean()), 4),
            "hot_cold_stability_p05": round(float(np.percentile(agree, 5)), 4),
            "hot_cold_stability_min_layer_mean": round(float(agree.mean(axis=1).min()), 4),
            "experts_stable_ge_0p95_fraction": round(float((agree >= 0.95).mean()), 4),
            "experts_stable_ge_0p90_fraction": round(float((agree >= 0.90).mean()), 4),
            **tokens_needed(counts, n_tok),
        },
    }


def _verdict(s: dict) -> tuple[bool, str]:
    ok = s["experts_stable_ge_0p95_fraction"] >= 0.90 and s["never_routed_fraction"] <= 0.01
    if ok:
        return True, ("The median hot/cold split is stable under token resampling and essentially "
                      "every expert is exercised. Safe to allocate bits per expert.")
    return False, (
        f"NOT trustworthy for PER-EXPERT bit allocation. Only "
        f"{s['experts_stable_ge_0p95_fraction']*100:.1f} percent of (layer, expert) hot/cold "
        f"assignments survive token resampling at 95 percent, and "
        f"{s['never_routed_fraction']*100:.1f} percent of experts are never routed at all on "
        f"{s['n_tokens']} tokens (expected count per expert is only "
        f"{s['expected_count_per_expert']}). A stable median partition needs roughly "
        f"{s['tokens_needed_for_90pct_of_experts']} tokens (median expert: "
        f"{s['tokens_needed_median_expert']}). Until then use the routing signal only as a COARSE "
        f"aggregate (quartile bands, or the lower-variance softmax_mass which has no zeros), and "
        f"gate any allocation on a real parent-vs-packed forward.")


# ── report ───────────────────────────────────────────────────────────────────────────────────
def build_report(raw: dict, st: dict) -> dict:
    counts, hot = st["counts"], st["hot"]
    layers = []
    for L in range(raw["n_layers"]):
        c = counts[L]
        q = np.percentile(c, [25, 50, 75])
        layers.append({
            "layer": L,
            "top8_count": c.tolist(),
            "topk_mass": [round(float(v), 6) for v in st["topk_mass"][L]],
            "softmax_mass": [round(float(v), 6) for v in st["soft_mass"][L]],
            "count_quartiles": {"q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])},
            "hot": np.nonzero(hot[L])[0].tolist(),
            "cold": np.nonzero(~hot[L])[0].tolist(),
            "coldest_quartile": np.nonzero(c <= q[0])[0].tolist(),
            "hottest_quartile": np.nonzero(c >= q[2])[0].tolist(),
            "never_routed": np.nonzero(c == 0)[0].tolist(),
            "hot_cold_stability_mean": round(float(st["agree"][L].mean()), 4),
        })
    trusted, verdict = _verdict(st["sampling"])
    rep = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "models/qwen3-235b-a22b (full resident, 118 shards, real forward)",
        "method": "real qwen_real_forward residual stream over the 6-prompt campaign holdout; "
                  "router logits softmaxed over all 128 experts then top-8 (norm_topk_prob); "
                  "lm_head and final norm skipped (routing does not read them)",
        "calibration_prompts": raw["prompts"],
        "n_tokens_calibrated_on": raw["n_tokens"],
        "n_layers": raw["n_layers"], "n_experts": raw["n_experts"], "top_k": raw["top_k"],
        "wall_seconds": raw["wall_seconds"],
        "sampling_error": st["sampling"],
        "trustworthy_for_per_expert_allocation": trusted,
        "trust_verdict": verdict,
        "partition_definition": "hot = top half of the 128 experts by top8_count within the layer "
                                "(ties broken by softmax_mass); cold = the other half. Exactly 64 "
                                "hot and 64 cold per layer.",
        "layers": layers,
    }
    rep["sha256"] = hashlib.sha256(
        json.dumps(rep, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rep


# ── consumer API ─────────────────────────────────────────────────────────────────────────────
_CACHE: dict | None = None


def load_partition(layer: int, path: str | os.PathLike[str] = REPORT) -> dict:
    """-> {"hot": set[int], "cold": set[int]} for one layer. For the campaign allocator."""
    global _CACHE
    if _CACHE is None or _CACHE.get("_path") != str(path):
        _CACHE = json.loads(Path(path).read_text()) | {"_path": str(path)}
    row = _CACHE["layers"][int(layer)]
    if row["layer"] != int(layer):
        row = next(r for r in _CACHE["layers"] if r["layer"] == int(layer))
    return {"hot": set(row["hot"]), "cold": set(row["cold"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real Qwen3-235B routing-frequency calibration.")
    ap.add_argument("--layers", type=int, default=None, help="bound the pass to the first N layers")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    ap.add_argument("--out", default=str(REPORT))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--calib-tokens", type=int, default=None,
                    help="use the frozen disjoint calibration corpus at >= N tokens (>=1000 advised)")
    args = ap.parse_args(argv)

    raw = collect(max_layers=args.layers, progress=not args.quiet,
                  calib_tokens=args.calib_tokens)
    rep = build_report(raw, analyse(raw, bootstrap=args.bootstrap))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({"sealed": str(out), "sha256": rep["sha256"],
                      **rep["sampling_error"], "trust_verdict": rep["trust_verdict"]}, indent=2))
    return 0


def demo() -> None:
    """Self-check on synthetic selections: partition, bootstrap and never-routed accounting."""
    n_l, n_tok, k, n_e = 3, 88, 8, 128
    rng = np.random.default_rng(0)
    # expert e picked with probability proportional to (e+1): a strong, unambiguous hot/cold signal
    p = np.arange(1, n_e + 1, dtype=np.float64); p /= p.sum()
    sel = np.stack([rng.choice(n_e, size=(n_tok, k), p=p) for _ in range(n_l)]).astype(np.int16)
    raw = {"sel_idx": sel, "sel_w": np.full((n_l, n_tok, k), 1.0 / k, dtype=np.float32),
           "soft_mass": np.tile(p * n_tok, (n_l, 1)), "n_experts": n_e, "top_k": k,
           "n_layers": n_l, "n_tokens": n_tok, "prompts": [], "wall_seconds": 0.0}
    st = analyse(raw, bootstrap=200)
    assert st["counts"].sum() == n_l * n_tok * k, "counts must conserve every router decision"
    assert st["hot"].sum(axis=1).tolist() == [n_e // 2] * n_l, "exactly half the experts are hot"
    assert st["counts"][0][st["hot"][0]].sum() > st["counts"][0][~st["hot"][0]].sum()
    assert 0.0 <= st["sampling"]["never_routed_fraction"] <= 1.0
    assert st["sampling"]["hot_cold_stability_mean"] <= 1.0
    # the low-probability tail must be the part that is unstable / never routed
    assert st["hot"][0][-1] and not st["hot"][0][0], "expert 127 hot, expert 0 cold by construction"
    rep = build_report(raw, st)
    assert len(rep["sha256"]) == 64 and rep["layers"][0]["layer"] == 0
    assert len(rep["layers"][0]["hot"]) == n_e // 2
    print("demo ok:", st["sampling"]["never_routed_fraction"],
          st["sampling"]["hot_cold_stability_mean"])


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(main())
