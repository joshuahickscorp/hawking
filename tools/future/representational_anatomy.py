"""OI representational anatomy: how does this organism want to be represented?

S012 §11 requires this BEFORE quantization is attempted, not after it fails.
Everything here was validated on two specimens: it predicted the shared-basis
NR's collapse (ppl 27.77) from spectra alone, and the second specimen's answer
arrived in five minutes instead of hours.

Reads safetensors directly rather than loading a model, so a partially
downloaded specimen still yields an anatomy -- mlx_lm's loader refuses those
outright, and the spectra need one complete expert group, not a runnable model.
"""
from __future__ import annotations

import collections
import glob
import re
from typing import Any

import mlx.core as mx

# Thresholds from the two-specimen family prior. Stated so they can be falsified.
SHARING_LIVE_BELOW = 0.95      # cross-expert participation ratio
LOWRANK_LIVE_BELOW = 0.40      # within-expert participation ratio

_EXPERT_RE = re.compile(r"\.layers\.(\d+)\..*experts?\.(\d+)\.(\w+proj)\.weight$")


def _spectrum(X: "mx.array") -> dict:
    Xc = X - mx.mean(X, axis=0, keepdims=True)
    Gm = (Xc @ Xc.T).astype(mx.float32) / max(Xc.shape[1], 1)
    ev = mx.maximum(mx.sort(mx.linalg.eigvalsh(Gm, stream=mx.cpu))[::-1], 0.0)
    tot = float(mx.sum(ev))
    part = ev / max(tot, 1e-30)
    ent = float(-mx.sum(mx.where(part > 0, part * mx.log(mx.maximum(part, 1e-30)), 0.0)))
    cum = mx.cumsum(ev) / max(tot, 1e-30)
    n = int(X.shape[0])
    p = float(mx.exp(mx.array(ent)).item())
    return {"n": n, "participation": round(p, 2), "ratio": round(p / n, 4),
            "rank_90": int(mx.sum(cum < 0.90).item()) + 1,
            "top1_share": round(float(ev[0] / max(tot, 1e-30)), 4)}


def _effective_rank(W: "mx.array") -> dict:
    s = mx.linalg.svd(W.astype(mx.float32), compute_uv=False, stream=mx.cpu)
    s2 = s * s
    tot = float(mx.sum(s2))
    part = s2 / max(tot, 1e-30)
    ent = float(-mx.sum(mx.where(part > 0, part * mx.log(mx.maximum(part, 1e-30)), 0.0)))
    cum = mx.cumsum(s2) / max(tot, 1e-30)
    full = int(min(W.shape))
    p = float(mx.exp(mx.array(ent)).item())
    return {"shape": list(W.shape), "full_rank": full,
            "participation": round(p, 1), "ratio": round(p / full, 4),
            "rank_90": int(mx.sum(cum < 0.90).item()) + 1}


def hypotheses(anatomy: dict) -> list[str]:
    """What the numbers license, in plain terms. This is the OI deliverable."""
    out = []
    ce = anatomy.get("cross_expert") or []
    if ce:
        r = min(x["ratio"] for x in ce)
        if r >= SHARING_LIVE_BELOW:
            out.append(f"cross-expert sharing is DEAD (ratio {r:.4f} >= {SHARING_LIVE_BELOW}): "
                       "skip shared basis, common+delta, clustering, alignment")
        else:
            out.append(f"cross-expert sharing is LIVE (ratio {r:.4f} < {SHARING_LIVE_BELOW}): "
                       "a shared basis may pay -- this specimen contradicts the family prior")
    we = anatomy.get("within_expert")
    if we:
        if we["ratio"] >= LOWRANK_LIVE_BELOW:
            out.append(f"low-rank factorisation is DEAD (ratio {we['ratio']:.4f} >= "
                       f"{LOWRANK_LIVE_BELOW}): a rank-{we['rank_90']} factor costs more than dense")
        else:
            out.append(f"low-rank is LIVE (ratio {we['ratio']:.4f}): rank-{we['rank_90']} may pay")
    cl = anatomy.get("cross_layer")
    if cl:
        verdict = "DEAD" if cl["ratio"] >= SHARING_LIVE_BELOW else "LIVE"
        out.append(f"cross-layer sharing is {verdict} (ratio {cl['ratio']:.4f})")
    if all("DEAD" in h for h in out) and out:
        out.append("NO LINEAR STRUCTURE AVAILABLE: go non-linear (generated coefficients, "
                   "procedural reconstruction, router-conditioned representation) or accept "
                   "quantization as the ceiling for this organism")
    return out


def anatomy_from_safetensors(snapshot: str, layer: int = 0, max_proj: int = 3) -> dict:
    shards = sorted(glob.glob(snapshot.rstrip("/") + "/*.safetensors"))
    groups: dict[str, dict[int, tuple]] = collections.defaultdict(dict)
    stacked: list[tuple[str, Any]] = []
    cache: dict[str, Any] = {}

    def load(f):
        if f not in cache:
            cache[f] = mx.load(f)
        return cache[f]

    for f in shards:
        for k, v in load(f).items():
            m = _EXPERT_RE.search(k)
            if m and int(m.group(1)) == layer:
                groups[m.group(3)][int(m.group(2))] = (f, k)
            elif v.ndim == 3 and v.shape[0] >= 8 and "expert" in k:
                stacked.append((k, v))

    out: dict[str, Any] = {"snapshot": snapshot, "layer": layer,
                           "storage": "per-expert" if groups else "stacked",
                           "cross_expert": []}
    if groups:
        for proj, d in sorted(groups.items())[:max_proj]:
            ids = sorted(d)
            X = mx.stack([load(d[e][0])[d[e][1]].astype(mx.float32).reshape(-1) for e in ids])
            r = _spectrum(X); r["tensor"] = proj
            out["cross_expert"].append(r)
        f, k = groups[sorted(groups)[0]][0]
        out["within_expert"] = _effective_rank(load(f)[k].astype(mx.float32))
    elif stacked:
        for k, v in stacked[:max_proj]:
            r = _spectrum(v.astype(mx.float32).reshape(v.shape[0], -1))
            r["tensor"] = k.split(".")[-2]
            out["cross_expert"].append(r)
        out["within_expert"] = _effective_rank(stacked[0][1][0].astype(mx.float32))
    out["hypotheses"] = hypotheses(out)
    return out
