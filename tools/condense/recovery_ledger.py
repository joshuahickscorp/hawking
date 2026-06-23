#!/usr/bin/env python3.12
"""Per-layer^* RECOVERY LEDGER for the condense doctor.

(*) "per-layer" in the doctor's sense = per recovery METHOD applied across the
    whole model at a given bit budget. The audit JSONL measures output-space
    degradation for each method (RHT -> AWQ -> residual ...) at a comparable
    effective bpw; this tool reads those runs and computes, for each bit tier,
    how much each method recovers RELATIVE TO THE PREVIOUS one (the delta), then
    flags the tier with the most REMAINING headroom (largest residual degradation
    after the best available method). That is where the next doctor effort pays.

INPUT (JSONL, one object per line; tolerant of partial/mixed schemas):
  {"model": "0.5B", "config": "3-AWQ", "eff_bpw": 3.654, "ppl": 42.07, "degr_pct": 14.6}
  - config encodes the method as "<bits>-<METHOD>[.suffix]" (RHT, AWQ, ...)
    or "res<b1>+<b2>" (residual). "f16" is the lossless anchor.
  - degr_pct is preferred; if absent but ppl + an f16 ppl exist, it is derived.
  Files like /tmp/audit_05b.jsonl and reports/condense/*.jsonl are the typical inputs.

The method ladder (recovery order) is RHT (baseline codec) -> AWQ -> residual.
Within a bit tier we sort configs by that ladder and report each step's delta
(how many degradation-points it removed vs the previous step).

Usage:
  python3.12 tools/condense/recovery_ledger.py [file1.jsonl ...] [-o out.md]
  # default inputs: /tmp/audit_05b.jsonl + reports/condense/*.jsonl (whatever exists)
  # default output: stdout (markdown)
"""
import sys, os, re, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# recovery-ladder rank: lower = earlier/weaker recovery, higher = stronger
LADDER = {"RHT": 0, "AWQ": 1, "RES": 2}


def method_of(config):
    """Map a config string to (bit_tier_label, method_name, ladder_rank)."""
    if config == "f16":
        return ("f16", "f16", -1)
    m = re.match(r"res(\d+)\+(\d+)", config)
    if m:
        b1, b2 = int(m.group(1)), int(m.group(2))
        return (f"{b1}-bit(+{b2})", "RES", LADDER["RES"])
    m = re.match(r"(\d+)-([A-Za-z]+)", config)
    if m:
        bits, meth = int(m.group(1)), m.group(2).upper()
        rank = LADDER.get(meth, 1)  # unknown methods sort with AWQ (mid)
        return (f"{bits}-bit", meth, rank)
    # ppl_bench/sweep labels: "tq3"/"tq2" = raw STRAND codec (RHT baseline) at N bits
    m = re.match(r"tq(\d+)", config, re.IGNORECASE)
    if m:
        return (f"{int(m.group(1))}-bit", "RHT", LADDER["RHT"])
    return (config, config.upper(), 1)


def load(paths):
    rows, f16_ppl = [], {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p, errors="ignore"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            # audit format uses "config"; ppl_bench/sweep format uses "label"
            if "config" not in r:
                if "label" in r:
                    r["config"] = r["label"]
                else:
                    continue
            model = r.get("model", "?")
            if r["config"] == "f16" and r.get("ppl") is not None:
                f16_ppl[model] = r["ppl"]
            rows.append(r)
    # derive degr_pct where missing but ppl + an f16 anchor exist
    for r in rows:
        if r.get("degr_pct") is None and r.get("ppl") is not None:
            base = f16_ppl.get(r.get("model", "?"))
            if base:
                r["degr_pct"] = round((r["ppl"] / base - 1) * 100, 2)
    return rows, f16_ppl


def best_row(group):
    """Lowest-degradation row in a list (skips rows with no degr_pct)."""
    cand = [r for r in group if r.get("degr_pct") is not None]
    return min(cand, key=lambda r: r["degr_pct"]) if cand else None


def render(rows, f16_ppl):
    out = []
    models = sorted({r.get("model", "?") for r in rows})
    out.append("# Condense recovery ledger\n")
    out.append("Per bit-tier, each doctor method's degradation vs f16 and the "
               "**delta** (points recovered) vs the previous method on the "
               "ladder RHT -> AWQ -> residual. The tier with the largest "
               "remaining degradation after the best method is the headroom.\n")

    headroom = []  # (model, tier_label, remaining_degr, best_config)
    for model in models:
        mrows = [r for r in rows if r.get("model", "?") == model and r["config"] != "f16"]
        if not mrows:
            continue
        base = f16_ppl.get(model)
        out.append(f"\n## {model}" + (f"  (f16 ppl {base:.2f})" if base else ""))
        # group by bit tier
        tiers = {}
        for r in mrows:
            tier, meth, rank = method_of(r["config"])
            r["_tier"], r["_meth"], r["_rank"] = tier, meth, rank
            tiers.setdefault(tier, []).append(r)

        # order tiers by their representative bpw (high bpw first)
        def tier_bpw(t):
            vals = [x.get("eff_bpw") for x in tiers[t] if x.get("eff_bpw") is not None]
            return max(vals) if vals else 0
        for tier in sorted(tiers, key=tier_bpw, reverse=True):
            # collapse method variants (e.g. AWQ alpha sweep 3-AWQ.25/.75) to the
            # BEST row per method, so the delta chain compares distinct recovery
            # stages (RHT -> AWQ -> residual), not parameter sweeps within a method.
            by_meth = {}
            for r in tiers[tier]:
                cur = by_meth.get(r["_meth"])
                if cur is None or (r.get("degr_pct") is not None
                                   and (cur.get("degr_pct") is None
                                        or r["degr_pct"] < cur["degr_pct"])):
                    by_meth[r["_meth"]] = r
            grp = sorted(by_meth.values(), key=lambda r: r["_rank"])
            out.append(f"\n### {tier}")
            out.append("| method | config | eff bpw | degr vs f16 | recovered vs prev |")
            out.append("|---|---|--:|--:|--:|")
            prev = None
            for r in grp:
                d = r.get("degr_pct")
                bpw = r.get("eff_bpw")
                bpw_s = f"{bpw:.3f}" if bpw is not None else "?"
                d_s = f"+{d:.2f}%" if d is not None else "?"
                if prev is not None and d is not None and prev.get("degr_pct") is not None:
                    delta = prev["degr_pct"] - d
                    sign = "+" if delta > 0 else ""
                    delta_s = f"{sign}{delta:.2f} pts"
                else:
                    delta_s = "— (baseline)"
                out.append(f"| {r['_meth']} | {r['config']} | {bpw_s} | {d_s} | {delta_s} |")
                if d is not None:
                    prev = r
            b = best_row(grp)
            if b is not None:
                headroom.append((model, tier, b["degr_pct"], b["config"]))

    # headroom summary: largest remaining degradation after best method per tier
    if headroom:
        out.append("\n## Where the most headroom remains")
        out.append("Sorted by remaining degradation after the best available method "
                   "in each tier — the top rows are where the next doctor lever "
                   "(stronger residual, KD, mixed-precision) should aim.\n")
        out.append("| model | tier | best method so far | remaining degr |")
        out.append("|---|---|---|--:|")
        for model, tier, d, cfg in sorted(headroom, key=lambda x: -x[2]):
            out.append(f"| {model} | {tier} | {cfg} | +{d:.2f}% |")
        top = max(headroom, key=lambda x: x[2])
        out.append(f"\n**Biggest headroom: {top[0]} {top[1]}** — best so far "
                   f"`{top[3]}` still +{top[2]:.2f}% over f16.")
    else:
        out.append("\n_(no rows with degradation data — check input JSONL paths)_")
    return "\n".join(out) + "\n"


def main():
    args = sys.argv[1:]
    out_path = None
    if "-o" in args:
        i = args.index("-o")
        out_path = args[i + 1]
        args = args[:i] + args[i + 2:]
    paths = args or ([os.path.join("/tmp", "audit_05b.jsonl")]
                     + glob.glob(os.path.join(ROOT, "reports", "condense", "*.jsonl")))
    rows, f16_ppl = load(paths)
    md = render(rows, f16_ppl)
    if out_path:
        open(out_path, "w").write(md)
        print(f"# wrote {out_path} ({len(rows)} rows from {len(paths)} file(s))",
              file=sys.stderr)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
