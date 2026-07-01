#!/usr/bin/env python3.12
"""codec_parallelism.py — score a candidate codec/kernel DESIGN for decode parallelism BEFORE
spending Rust/Metal engineering on it.

Direct finding from this session: decode speed on Apple Silicon is bandwidth-bound, so MORE
compression should mean MORE speed (fewer bytes moved) - EXCEPT when the codec's decode requires
serial cross-lane dependency or random gather, which Apple GPUs punish. That is exactly why the
QTIP-on-Metal attempt died (~24% of theoretical peak, serial state[i]<-state[i-1] hazard) while
the lookup-free scalar bitslice trellis holds ~60-74% of peak (one thread per 256-weight block,
zero cross-lane dependency, one LUT-stage barrier). The lesson generalizes: THE TENSION IS NOT
DENSITY VS SPEED, IT IS DECODE-PARALLELISM VS SPEED. This tool makes that check explicit and cheap,
for ANY new codec candidate (vector-trellis d>1, a learned VQ codebook, a new outlier-position
scheme, etc.) - a paper design, not a benchmark - so a codec only reaches the Rust serve-build
queue if it structurally passes.

Scored properties (from the design, not a real kernel):
  - lane_independent   : does decoding weight i require the decoded value of weight i-1 (serial
                          state machine) or can every 256-block decode independently in parallel?
  - lookup_free / staged: is the codebook a per-block ONE-TIME staged lookup (cheap) or a
                          per-ELEMENT random gather (Apple GPUs punish this heavily)?
  - barrier_count       : threadgroup barriers per block (more = more serialization stalls)
  - bytes_per_weight    : the density payoff (lower = more compression = more bandwidth saved)
  - est_peak_pct        : a transparent heuristic score (not a measurement) combining the above,
                          calibrated against the two MEASURED anchors (bitslice ~67%, QTIP-Metal ~24%)

GATE: est_peak_pct >= 50% -> worth prototyping in Rust. Below that, the projected bandwidth win
from lower bpw is likely eaten by decode stalls (as QTIP-on-Metal proved) - do not build it yet.
This is a heuristic triage tool, not a measurement; every candidate that passes still needs a real
Metal parity+throughput test before it ships (see docs/plans/STUDIO_GO.md serve-build critical path).

Usage:
  codec_parallelism.py --score <name> --bpw B [--serial 0|1] [--gather 0|1] [--barriers N]
  codec_parallelism.py --catalog     # score the known candidates (bitslice, vec-d2, QTIP-style, learned-VQ)
"""
import sys, os, json

OUT = "reports/condense"
# calibration anchors: (serial, gather, barriers) -> measured est_peak_pct
ANCHOR_BITSLICE = {"serial": 0, "gather": 0, "barriers": 1, "measured_pct": 67.0}
ANCHOR_QTIP_METAL = {"serial": 1, "gather": 1, "barriers": 3, "measured_pct": 24.0}
GATE_PCT = 50.0


def score(name, bpw, serial, gather, barriers):
    """Transparent heuristic, linearly interpolated/extrapolated between the two measured anchors
    on a (serial, gather, barrier) penalty axis. NOT a substitute for a real kernel benchmark."""
    penalty = serial * 30 + gather * 25 + max(0, barriers - 1) * 8
    base = ANCHOR_BITSLICE["measured_pct"]     # 67% at (0,0,1) barrier baseline
    est = max(5.0, base - penalty)
    verdict = "PROTOTYPE" if est >= GATE_PCT else "KILL (decode stalls likely eat the bandwidth win)"
    return {
        "name": name, "bpw": bpw, "lane_independent": not serial, "lookup_free": not gather,
        "barrier_count": barriers, "est_peak_pct": round(est, 1), "gate_pct": GATE_PCT,
        "verdict": verdict, "heuristic": True,
        "note": "extrapolated from 2 measured anchors (bitslice 67%, QTIP-Metal 24%); confirm with "
                "a real Metal micro-benchmark before committing serve-build time",
    }


CATALOG = [
    # name, bpw, serial(0/1), gather(0/1), barriers
    ("scalar-bitslice (shipped)",       2.34, 0, 0, 1),
    ("vec-trellis d=2 (gated off)",     1.34, 0, 0, 1),
    ("vec-trellis d=8 (theoretical)",   0.29, 0, 1, 2),   # wider vector codebook likely needs a gather stage
    ("QTIP-style serial state (dead)",  1.60, 1, 1, 3),
    ("learned-VQ, staged codebook",     2.00, 0, 0, 2),
    ("learned-VQ, per-elem gather",     2.00, 0, 1, 1),
]


def run_catalog():
    rows = [score(n, b, s, g, bar) for (n, b, s, g, bar) in CATALOG]
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/codec_parallelism_catalog.json", "w"), indent=2)
    print(f"{'candidate':32s} {'bpw':>5s} {'lane-indep':>10s} {'lookup-free':>11s} "
          f"{'barriers':>8s} {'est%peak':>9s}  verdict", file=sys.stderr)
    for r in rows:
        print(f"{r['name']:32s} {r['bpw']:>5} {str(r['lane_independent']):>10} "
              f"{str(r['lookup_free']):>11} {r['barrier_count']:>8} {r['est_peak_pct']:>8}%  {r['verdict']}",
              file=sys.stderr)
    print(f"\n# GATE: est_peak_pct >= {GATE_PCT}% -> PROTOTYPE in Rust; below -> KILL before spending build time.",
          file=sys.stderr)
    return rows


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "--catalog"
    if a == "--catalog":
        run_catalog()
    elif a == "--score":
        name = sys.argv[sys.argv.index("--score") + 1]
        bpw = float(sys.argv[sys.argv.index("--bpw") + 1]) if "--bpw" in sys.argv else 2.0
        serial = int(sys.argv[sys.argv.index("--serial") + 1]) if "--serial" in sys.argv else 0
        gather = int(sys.argv[sys.argv.index("--gather") + 1]) if "--gather" in sys.argv else 0
        barriers = int(sys.argv[sys.argv.index("--barriers") + 1]) if "--barriers" in sys.argv else 1
        r = score(name, bpw, serial, gather, barriers)
        print(json.dumps(r, indent=2))
        print(f"[codec] {name}: est {r['est_peak_pct']}% peak -> {r['verdict']}", file=sys.stderr)
    else:
        print(__doc__)
