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


def _separation(rs, key):
    """Between-family spread against the worst within-family spread, on one axis."""
    fams: dict[str, list] = {}
    for r in rs:
        v = r.get(key)
        if v:
            fams.setdefault(r["family"] or "unknown", []).append(v)
    if not fams:
        return None
    per = {f: {"n": len(v), "median": round(statistics.median(v), 2),
               "spread_pct": (round((max(v) - min(v)) / statistics.median(v) * 100, 1)
                              if len(v) > 1 else None)}
           for f, v in sorted(fams.items())}
    meds = [d["median"] for d in per.values()]
    between = ((max(meds) - min(meds)) / statistics.median(meds) * 100
               if len(meds) > 1 else None)
    multi = {f: d for f, d in per.items() if d["n"] > 1}
    within = max((d["spread_pct"] for d in multi.values()), default=None)
    # One family with two bodies is not a within-family estimate, it is a
    # single pair. The rule used to accept it and reported SEPARATES on data
    # where four of five families had n=1 and the whole spread came from one
    # outlier. Require the within term to rest on at least TWO families before
    # a separation claim is allowed.
    if not multi:
        verdict = ("UNDECIDABLE: no family has two bodies, so there is no within-family "
                   "spread to compare against")
    elif len(multi) < 2:
        only = next(iter(multi))
        verdict = (f"UNDECIDABLE: only {only} has more than one body, so the within-family "
                   f"term is a single pair ({within}%), not an estimate. between={between:.0f}%")
    elif between is not None and within is not None and between > 2 * within:
        verdict = f"SEPARATES: between {between:.0f}% vs worst within {within:.0f}%"
    else:
        verdict = f"DOES NOT SEPARATE: between {between:.0f}% vs worst within {within}%"
    return {"verdict": verdict, "per_family": per,
            "between_family_spread_pct": between,
            "worst_within_family_spread_pct": within,
            "range": [round(min(meds), 2), round(max(meds), 2)] if meds else None}


def report(rs):
    """Decode and prefill judged by the SAME rule in _separation().

    report() used to carry its own copy of the verdict logic. Tightening the
    rule in one place then left the other loose, which is how a claim can
    survive a fix -- so there is one authority now and report() defers to it.

    Decode is memory-bandwidth-bound and its ratio tends toward a host
    constant; prefill is compute-bound and is where an architecture prior could
    actually live. Report both and let the axis with separation earn it.
    """
    if not rs:
        return {"verdict": "NO BODY IS MEASURED ON BOTH AXES YET", "n": 0}
    dec = _separation(rs, "decode_ratio")
    pre = _separation(rs, "prefill_ratio")
    return {"verdict": dec["verdict"], "n": len(rs),
            "per_family": dec["per_family"],
            "between_family_spread_pct": dec["between_family_spread_pct"],
            "worst_within_family_spread_pct": dec["worst_within_family_spread_pct"],
            "decode": dec, "prefill": pre,
            "why_two_axes": ("decode is memory-bandwidth-bound and its ratio tends toward a "
                             "host constant; prefill is compute-bound and is where an "
                             "architecture prior can actually live"),
            "bodies": sorted(rs, key=lambda r: -r["decode_ratio"])}


def _demo():
    """Self-check: the verdict must actually depend on the numbers."""
    # two families, three bodies each: the within term rests on both
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

    # one multi-body family beside singletons is NOT enough for a claim
    thin = [dict(tight[0]), dict(tight[1]),
            {"slug": "c0", "family": "z", "gib": 1, "gpu_decode": 100, "cpu_decode": 2,
             "decode_ratio": 50.0, "prefill_ratio": None},
            {"slug": "d0", "family": "w", "gib": 1, "gpu_decode": 100, "cpu_decode": 3,
             "decode_ratio": 33.0, "prefill_ratio": None}]
    v = report(thin)["verdict"]
    assert v.startswith("UNDECIDABLE") and "single pair" in v, v
    assert report([])["n"] == 0

    # The two axes must be judged INDEPENDENTLY. The live data has a tight
    # decode band (11.6-13.8x, a host constant) sitting beside a 7x prefill
    # spread, so a tool that judged only decode would report "no family prior
    # exists" while the prefill signal was already there.
    mixed = []
    for i in range(3):                      # decode flat, prefill separated
        mixed.append({"slug": f"x{i}", "family": "x", "gib": 1, "gpu_decode": 100,
                      "cpu_decode": 8, "decode_ratio": 12.0 + i * 0.1,
                      "prefill_ratio": 7.0 + i * 0.1})
        mixed.append({"slug": f"y{i}", "family": "y", "gib": 1, "gpu_decode": 100,
                      "cpu_decode": 8, "decode_ratio": 12.0 + i * 0.1,
                      "prefill_ratio": 50.0 + i * 0.1})
    rep = report(mixed)
    assert rep["decode"]["verdict"].startswith("DOES NOT SEPARATE"), rep["decode"]
    assert rep["prefill"]["verdict"].startswith("SEPARATES"), rep["prefill"]
    # and the reverse, so neither axis is hard-coded to an answer
    flipped = [dict(r, decode_ratio=r["prefill_ratio"], prefill_ratio=r["decode_ratio"])
               for r in mixed]
    rep2 = report(flipped)
    assert rep2["decode"]["verdict"].startswith("SEPARATES"), rep2["decode"]
    assert rep2["prefill"]["verdict"].startswith("DOES NOT SEPARATE"), rep2["prefill"]
    print("cpu_gpu_ratio: verdict tracks the data (separates / overlaps / undecidable / empty)")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        _demo()
    else:
        print(json.dumps(report(rows()), indent=2))
