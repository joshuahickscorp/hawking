"""Does the CPU:GPU decode ratio separate architecture families?

A body measured on both axes gives a ratio that is dimensionless -- it divides
out the body's size and the host's absolute speed, leaving how much this
architecture depends on the GPU. If the ratio clusters by family, it is a cheap
physical prior: you can predict a new body's GPU benefit from its family before
spending any GPU time on it.

This is a screen, not a law. It reports the spread WITHIN each family next to
the gap BETWEEN families, because a family-level prior is only worth having if
between exceeds within. With one or two bodies in a family there is no within,
and the honest output says so rather than quoting a mean of one.
"""
from __future__ import annotations
import json
import statistics
import sys
from pathlib import Path

LEDGER = Path("receipts/future/G034_ODYSSEY_LEDGER.json")


def rows(path: Path = LEDGER):
    doc = json.loads(path.read_text())
    out = []
    for s in doc.get("specimens") or []:
        ax = s.get("axes") or {}
        g, c = ax.get("gpu") or {}, ax.get("cpu") or {}
        if g.get("state") != "MEASURED" or c.get("state") != "MEASURED":
            continue
        gv, cv = g.get("value") or {}, c.get("value") or {}
        gd, cd = gv.get("decode_tps_median"), cv.get("decode_tps_median")
        gp, cp = gv.get("prefill_tps_median"), cv.get("prefill_tps_median")
        if not (gd and cd):
            continue
        out.append({"slug": s["slug"], "family": s.get("family"), "gib": s.get("gib"),
                    "gpu_decode": gd, "cpu_decode": cd,
                    "decode_ratio": gd / cd,
                    "prefill_ratio": (gp / cp) if (gp and cp) else None})
    return out


def report(rs):
    if not rs:
        return {"verdict": "NO BODY IS MEASURED ON BOTH AXES YET", "n": 0}
    fams: dict[str, list] = {}
    for r in rs:
        fams.setdefault(r["family"] or "unknown", []).append(r["decode_ratio"])
    per = {}
    for f, v in sorted(fams.items()):
        per[f] = {"n": len(v), "median": round(statistics.median(v), 2),
                  "spread_pct": (round((max(v) - min(v)) / statistics.median(v) * 100, 1)
                                 if len(v) > 1 else None)}
    multi = {f: d for f, d in per.items() if d["n"] > 1}
    meds = [d["median"] for d in per.values()]
    between = (max(meds) - min(meds)) / statistics.median(meds) * 100 if len(meds) > 1 else None
    within = max((d["spread_pct"] for d in multi.values()), default=None)
    if not multi:
        verdict = ("UNDECIDABLE: no family has two bodies, so there is no within-family "
                   "spread to compare the between-family gap against. Ratios below are "
                   "single observations, not family properties.")
    elif between is not None and within is not None and between > 2 * within:
        verdict = (f"SEPARATES: between-family spread {between:.0f}% exceeds the worst "
                   f"within-family spread {within:.0f}% by more than 2x")
    else:
        verdict = (f"DOES NOT SEPARATE: between-family {between:.0f}% vs worst "
                   f"within-family {within}% -- the families overlap")
    return {"verdict": verdict, "n": len(rs), "per_family": per,
            "between_family_spread_pct": between, "worst_within_family_spread_pct": within,
            "bodies": sorted(rs, key=lambda r: -r["decode_ratio"])}


def _demo():
    """Self-check: the verdict must actually depend on the numbers."""
    tight = [{"slug": f"a{i}", "family": "x", "gib": 1, "gpu_decode": 100,
              "cpu_decode": 10, "decode_ratio": 10 + i * 0.1, "prefill_ratio": None}
             for i in range(3)]
    tight += [{"slug": f"b{i}", "family": "y", "gib": 1, "gpu_decode": 100,
               "cpu_decode": 2, "decode_ratio": 50 + i * 0.1, "prefill_ratio": None}
              for i in range(3)]
    assert report(tight)["verdict"].startswith("SEPARATES"), report(tight)["verdict"]
    overlap = [dict(r, decode_ratio=10 + (i * 7) % 20) for i, r in enumerate(tight)]
    assert report(overlap)["verdict"].startswith("DOES NOT SEPARATE"), report(overlap)["verdict"]
    single = [dict(tight[0]), dict(tight[3])]
    assert report(single)["verdict"].startswith("UNDECIDABLE")
    assert report([])["n"] == 0
    print("cpu_gpu_ratio: verdict tracks the data (separates / overlaps / undecidable / empty)")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        print(json.dumps(report(rows()), indent=2))
