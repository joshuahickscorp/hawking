#!/usr/bin/env python3.12
"""Gravity 120B run: sub-bit F1 weight-space tournament on the REAL GPT-OSS expert weights.

SUPERSEDED baseline (2026-07-17): this is the naive weight-space PROXY run, sealed as
FORGE_BASELINE_NEGATIVE and superseded by tools/condense/gravity_forge.py. It is kept for
reference and MUST NOT be treated as authoritative. It measures relative-Frobenius reconstruction
error - a proxy, NOT the protected-capability contract - so it CANNOT name an Event Horizon or
emit a `gravity_event_horizon` claim (doctrine Section 2 / FORGE_BASELINE_NEGATIVE.must_not).
Notifications default OFF here for that reason.

Honest boundary: F1 measures weight reconstruction on the actual weights (a real error signal,
not a paper estimate). It is NOT the deployable sub-1-bit packer and NOT an HF capability-parity
claim; both remain gated. It brackets only the weight-space PROXY, never the capability Event
Horizon, using the proven per-expert loader.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gptoss_moe_runtime as rt  # noqa: E402
import succ_gravity as sg  # noqa: E402
import succ_gravity_policy as gp  # noqa: E402
from eco_common import seal_field, sealed, now_iso, atomic_write_json  # noqa: E402

RUN_SCHEMA = "hawking.gravity.run_120b.v1"
STATE_ROOT = Path("reports/condense/subbit_frontier/gravity_120b_run")

# survival thresholds on relative Frobenius error (real weights).
_SURVIVE, _DEGRADE = 0.15, 0.40


def _rel_error(w: np.ndarray, recon: np.ndarray) -> float:
    denom = float(np.linalg.norm(w)) or 1.0
    return float(np.linalg.norm(w - recon) / denom)


def _pick_dim(cols: int, bpw: float, k_cap: int) -> int:
    """Largest divisor of cols with 2**(bpw*dim) <= k_cap, so effective bpw tracks the target."""
    import math
    ceil_dim = max(1, int(math.log2(k_cap) / max(bpw, 1e-6)))
    divisors = [d for d in range(1, min(cols, 64) + 1) if cols % d == 0 and d <= ceil_dim]
    return max(divisors) if divisors else 1


def vq_reconstruct(w: np.ndarray, *, bpw: float, dim: int | None = None, iters: int = 5,
                   seed: int = 0, k_cap: int = 4096, fit_sample: int = 20000
                   ) -> tuple[np.ndarray, float]:
    """Vector-codebook (materially different family): tile W into dim-vectors (dim scaled so
    effective bpw tracks the target), k-means with K = min(2**(bpw*dim), k_cap) centroids fit
    on a bounded subsample, index each vector. Returns (reconstruction, effective_bpw)."""
    rows, cols = w.shape
    d = dim if dim is not None else _pick_dim(cols, bpw, k_cap)
    d = d if cols % d == 0 else 1
    K = int(min(k_cap, max(2, round(2 ** (bpw * d)))))
    v = w.reshape(-1, d)
    rng = np.random.default_rng(seed)
    fit = v if v.shape[0] <= fit_sample else v[rng.choice(v.shape[0], size=fit_sample, replace=False)]
    cb = fit[rng.choice(fit.shape[0], size=min(K, fit.shape[0]), replace=False)].copy()
    for _ in range(iters):
        idx = _blocked_assign(fit, cb).argmin(1)
        for k in range(cb.shape[0]):
            m = idx == k
            if m.any():
                cb[k] = fit[m].mean(0)
    recon = cb[_blocked_assign(v, cb).argmin(1)].reshape(rows, cols)
    eff_bpw = float(np.log2(cb.shape[0]) / d)
    return recon, eff_bpw


def _blocked_assign(v: np.ndarray, cb: np.ndarray, block: int = 8192) -> np.ndarray:
    out = np.empty((v.shape[0], cb.shape[0]), dtype=np.float32)
    for i in range(0, v.shape[0], block):
        chunk = v[i:i + block]
        out[i:i + block] = ((chunk[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    return out


def lowrank_reconstruct(w: np.ndarray, *, bpw: float) -> tuple[np.ndarray, float]:
    """Low-rank (materially different family): SVD rank-r sized to the target effective bpw
    with fp16 factors. Effective bpw = r*(rows+cols)*16/(rows*cols)."""
    rows, cols = w.shape
    r = max(1, int((bpw * rows * cols) / (16 * (rows + cols))))
    r = min(r, min(rows, cols))
    u, s, vt = np.linalg.svd(w.astype(np.float32), full_matrices=False)
    recon = (u[:, :r] * s[:r]) @ vt[:r]
    eff_bpw = float(r * (rows + cols) * 16 / (rows * cols))
    return recon.astype(np.float32), eff_bpw


def _verdict(err: float) -> str:
    return "survives" if err < _SURVIVE else ("degraded" if err < _DEGRADE else "collapse")


def f1_tournament(w: np.ndarray, rate: Fraction, *, sample_rows: int = 512) -> dict[str, Any]:
    """Run the two materially-different families at the target rate on a real weight.
    Samples up to sample_rows of the matrix for a fast, still-real reconstruction verdict."""
    if sample_rows and w.shape[0] > sample_rows:
        w = w[:sample_rows]
    bpw = float(rate)
    vq_recon, vq_bpw = vq_reconstruct(w, bpw=bpw)
    lr_recon, lr_bpw = lowrank_reconstruct(w, bpw=bpw)
    vq_err, lr_err = _rel_error(w, vq_recon), _rel_error(w, lr_recon)
    families = {
        "vector_codebook": {"class": "vector_codebook", "effective_bpw": round(vq_bpw, 4),
                            "rel_error": round(vq_err, 5), "verdict": _verdict(vq_err)},
        "low_rank": {"class": "low_rank_factor", "effective_bpw": round(lr_bpw, 4),
                    "rel_error": round(lr_err, 5), "verdict": _verdict(lr_err)},
    }
    best = min(families.values(), key=lambda f: f["rel_error"])
    return {"rate": gp.rate_identity(rate), "families": families,
            "best_family": [k for k, v in families.items() if v is best][0],
            "best_rel_error": best["rel_error"], "rate_verdict": best["verdict"]}


def probe_rate(reader: rt.ProvenanceReader, rate: Fraction, *, blocks: list[int],
               experts_per_block: int) -> dict[str, Any]:
    """F1 across a bounded sample of real experts at one rate. Aggregates the verdict."""
    probes = []
    for b in blocks:
        for e in range(experts_per_block):
            ex = rt.load_expert(reader, b, e)
            for organ in ("mlp1", "mlp2"):
                t = f1_tournament(ex[organ], rate)
                t.update({"block": b, "expert": e, "organ": organ})
                probes.append(t)
    verdicts = [p["rate_verdict"] for p in probes]
    survive = verdicts.count("survives")
    collapse = verdicts.count("collapse")
    agg = ("survives" if survive >= 0.8 * len(verdicts)
           else "collapse" if collapse >= 0.5 * len(verdicts) else "degraded")
    mean_err = float(np.mean([p["best_rel_error"] for p in probes]))
    return {"rate": gp.rate_identity(rate), "aggregate_verdict": agg, "mean_best_rel_error": round(mean_err, 5),
            "n_probes": len(probes), "survives": survive, "collapse": collapse, "probes": probes}


def run(*, source: str = "scratch/staging/gpt-oss-120b.partial",
        manifest: str = rt.DEFAULT_MANIFEST, blocks: list[int] | None = None,
        experts_per_block: int = 2, rates: list[str] | None = None,
        notify: bool = False) -> dict[str, Any]:
    """Start the Gravity 120B run: inverted search over sub-bit rates, F1 on real experts,
    checkpointed + notified. Bounded by blocks/experts_per_block so it starts and runs; the
    same run expands under unbounded wall-clock by widening the sample."""
    reader = rt.ProvenanceReader(manifest)
    sample = reader.by_name.get("block.0.mlp.gate.weight")
    if sample is None or not Path(sample["shard_path"]).exists():
        raise RuntimeError("120B source shards absent; cannot run F1 on real weights")

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    blocks = blocks if blocks is not None else [0, 18, 35]
    # inverted search order over sub-bit rates: start aggressively low, ascend on collapse
    rate_list = [gp.parse_rate(r) for r in (rates or ["1/4", "1/3", "1/2", "11/20", "4/5"])]

    if notify:
        import succ_telegram as tg
        tg.emit("gravity_tournament_started", {"parent": "120B", "rate": "sub-bit sweep",
                                              "blocks": str(blocks), "experts": experts_per_block})

    per_rate: list[dict[str, Any]] = []
    first_survive: Fraction | None = None
    lower_fail: Fraction | None = None
    started = now_iso()
    for rate in rate_list:
        t0 = time.time()
        res = probe_rate(reader, rate, blocks=blocks, experts_per_block=experts_per_block)
        res["seconds"] = round(time.time() - t0, 1)
        per_rate.append(res)
        # checkpoint after each rate (resumable)
        atomic_write_json(STATE_ROOT / f"rate_{rate.numerator}_{rate.denominator}.json",
                          seal_field(res, "rate_sha256"))
        if notify:
            tg.emit("gravity_feasibility_completed",
                    {"parent": "120B", "rate": gp.rate_identity(rate)["label"],
                     "tier": "F1", "verdict": res["aggregate_verdict"],
                     "mean_err": res["mean_best_rel_error"]})
        # require true 'survives' (not merely 'degraded') before naming any proxy boundary
        if res["aggregate_verdict"] == "survives" and first_survive is None:
            first_survive = rate
        if res["aggregate_verdict"] == "collapse" and first_survive is not None and lower_fail is None:
            lower_fail = rate

    doc = {
        "schema": RUN_SCHEMA, "parent": "120B", "started_at": started, "finished_at": now_iso(),
        "source": source, "blocks_sampled": blocks, "experts_per_block": experts_per_block,
        "rates": [gp.rate_identity(r) for r in rate_list],
        "per_rate": [{k: v for k, v in r.items() if k != "probes"} for r in per_rate],
        # proxy-explicit names; this run cannot name an Event Horizon (weight-space proxy only)
        "f1_weight_proxy_first_survival_rate": gp.rate_identity(first_survive) if first_survive else None,
        "f1_weight_proxy_lower_fail_rate": gp.rate_identity(lower_fail) if lower_fail else None,
        "evidence_level": "F1_weight_proxy", "metric_is_proxy": True,
        "is_event_horizon": False, "authorizes_escape": False, "is_deployable_artifact": False,
        "superseded_by": "tools/condense/gravity_forge.py",
        "note": "SUPERSEDED baseline-negative. Weight-space reconstruction proxy on real 120B "
                "experts; NOT a capability claim, NOT the Event Horizon; no escape authorized.",
    }
    doc = seal_field(doc, "run_sha256")
    atomic_write_json(STATE_ROOT / "RUN.json", doc)
    if notify:
        # F1-proxy-scoped only; never the capability `gravity_event_horizon` kind
        r = doc["f1_weight_proxy_first_survival_rate"]
        tg.emit("gravity_feasibility_completed", {"parent": "120B", "tier": "F1_weight_proxy",
                "verdict": "baseline_negative",
                "first_survival_rate": r["label"] if r else "none-survived"})
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gravity 120B sub-bit F1 run on real experts.")
    ap.add_argument("--blocks", default="0,18,35")
    ap.add_argument("--experts-per-block", type=int, default=2)
    ap.add_argument("--rates", default="1/4,1/3,1/2,11/20,4/5")
    ap.add_argument("--notify", action="store_true",
                    help="opt in to F1-proxy feasibility Telegram (superseded baseline; off by default)")
    args = ap.parse_args(argv)
    doc = run(blocks=[int(b) for b in args.blocks.split(",")],
              experts_per_block=args.experts_per_block,
              rates=args.rates.split(","), notify=args.notify)
    print(json.dumps({k: v for k, v in doc.items() if k != "per_rate"}, indent=2, sort_keys=True, default=str))
    print("\nper-rate F1 verdicts:")
    for r in doc["per_rate"]:
        print(f"  {r['rate']['label']:>5s}  {r['aggregate_verdict']:9s}  mean_err={r['mean_best_rel_error']}  ({r['seconds']}s)")
    assert sealed(doc, "run_sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
