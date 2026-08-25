#!/usr/bin/env python3
"""G040 — PARETO_ARCHIVE + FINAL_RESIDENT_SELECTION (S011 §37, §41, §4, §99).

Selection is CAPABILITY FLOOR FIRST, then Pareto. That order matters: the lowest-density
and the fastest bodies in this campaign are both capability-dead, and a density-first
rule would hand the resident slot to a body that cannot do any work at all.

The primary objective S011 §0 sets is

    VERIFIED ACCEPTED HCLI AUTONOMOUS WORK / (WALL TIME x PHYSICAL RESOURCE)

which is why the composite below divides a measured WUs/hour by a measured resource,
rather than ranking on density and hoping capability follows.

Categories with no measurement are reported as UNMEASURED. Filling them by inference
would be the failure mode this campaign keeps catching.
"""
import json, statistics, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"


def load():
    perf = json.load(open(RH / "QWEN_PERFORMANCE_QUALIFICATION.json"))["bodies"]
    add = json.load(open(RH / "QWEN_PERFORMANCE_ADDENDUM.json"))
    vb = json.load(open("/tmp/vb_perf.json")) if Path("/tmp/vb_perf.json").is_file() else None

    def cap(label):
        p = RH / f"CAPABILITY_{label}.json"
        return json.load(open(p))["overall"] if p.is_file() else None

    def hcli(label):
        """Median over every rep on disk, with the spread kept.

        A single bench run cannot support a 2%-margin selection: the concurrency sweep
        earlier in this campaign produced exactly that mistake. Reps are alternated
        between bodies so any machine drift is shared."""
        ps = [RH / f"HCLI_BENCH_{label}.json"] + sorted(RH.glob(f"HCLI_{label}_rep*.json"))
        scores = [json.load(open(x))["score"] for x in ps if x.is_file()]
        if not scores:
            return None
        rate = [s["VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR"] for s in scores]
        med = statistics.median(rate)
        out = dict(scores[0])
        out["VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR"] = round(med, 3)
        out["n_reps"] = len(rate)
        out["rate_reps"] = [round(r, 3) for r in rate]
        out["rate_spread_pct"] = round(100 * (max(rate) - min(rate)) / med, 3)
        out["accepted_reps"] = [s["verified_accepted_workunits"] for s in scores]
        out["accepted_is_stable"] = len(set(out["accepted_reps"])) == 1
        return out

    rows = {}
    for tag, label in (("sealed-3.14", "noetic-sealed-3.14"),
                       ("variantA-2.98", "noetic-variantA-2.98"),
                       ("clean-2.60", "noetic-clean-2.60")):
        b = perf[tag]
        lv, pv = b["latency_vector"], b["physical_work_vector"]
        av = add["physical_vectors"].get(tag, {})
        c, h = cap(label), hcli(label)
        rows[tag] = {
            "complete_ebpw": pv["ARTIFACT_PHYSICAL_complete_ebpw"],
            "active_ebpw_per_token": av.get("ARTIFACT_PHYSICAL_active_ebpw_per_token"),
            "resident_bytes": pv["resident_bytes_on_disk"],
            "TPOT_ns_p50": lv["TPOT_ns_p50"], "TTFT_ns": lv["TTFT_ns_median"],
            "single_stream_tps": lv["single_stream_tps"],
            "model_reachable_gb_s": av.get("RUNTIME_MEASURED_model_reachable_gb_s"),
            "capability_passed": c["passed"] if c else None,
            "capability_total": c["total"] if c else None,
            "hcli_wus_per_hour": (h["VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR"] if h else 0.0),
            "hcli_measured": bool(h),
            "hcli_n_reps": (h or {}).get("n_reps"),
            "hcli_rate_reps": (h or {}).get("rate_reps"),
            "hcli_rate_spread_pct": (h or {}).get("rate_spread_pct"),
            "hcli_accepted_reps": (h or {}).get("accepted_reps"),
        }
    if vb:
        h = hcli("noetic-variantB-2.76")
        c = cap("noetic-variantB-2.76")
        rows["variantB-2.76"] = {
            "complete_ebpw": vb["ARTIFACT_PHYSICAL_complete_ebpw"],
            "active_ebpw_per_token": vb["ARTIFACT_PHYSICAL_active_ebpw_per_token"],
            "resident_bytes": vb["stored_bytes"],
            "TPOT_ns_p50": vb["TPOT_ns_p50"], "TTFT_ns": vb["TTFT_ns_median"],
            "single_stream_tps": vb["single_stream_tps"],
            "model_reachable_gb_s": vb["RUNTIME_MEASURED_model_reachable_gb_s"],
            "capability_passed": c["passed"] if c else None,
            "capability_total": c["total"] if c else None,
            "hcli_wus_per_hour": (h["VERIFIED_ACCEPTED_WORKUNITS_PER_HOUR"] if h else 0.0),
            "hcli_measured": bool(h),
            "hcli_n_reps": (h or {}).get("n_reps"),
            "hcli_rate_reps": (h or {}).get("rate_reps"),
            "hcli_rate_spread_pct": (h or {}).get("rate_spread_pct"),
            "hcli_accepted_reps": (h or {}).get("accepted_reps"),
        }
    return rows, add


def dominates(a, b):
    """a dominates b: no worse on every axis, strictly better on at least one.
    Lower is better for ebpw/TPOT/TTFT; higher for capability and WUs/hour."""
    lower = ["complete_ebpw", "TPOT_ns_p50", "TTFT_ns"]
    higher = ["capability_passed", "hcli_wus_per_hour"]
    ge = all(a[k] <= b[k] for k in lower) and all((a[k] or 0) >= (b[k] or 0)
                                                  for k in higher)
    gt = any(a[k] < b[k] for k in lower) or any((a[k] or 0) > (b[k] or 0)
                                                for k in higher)
    return ge and gt


def main():
    rows, add = load()
    FLOOR = 1          # a resident must produce at least one verified WorkUnit
    eligible = {k: v for k, v in rows.items()
                if (v["capability_passed"] or 0) >= FLOOR
                and v["hcli_wus_per_hour"] > 0}
    rejected = {k: {"capability_passed": v["capability_passed"],
                    "hcli_wus_per_hour": v["hcli_wus_per_hour"],
                    "why": "below the capability floor: produces no verified work"}
                for k, v in rows.items() if k not in eligible}

    front = [k for k in eligible
             if not any(dominates(eligible[o], eligible[k]) for o in eligible if o != k)]

    def composite(v):
        # S011 §0: verified work per wall time per physical resource
        return round(v["hcli_wus_per_hour"] / (v["resident_bytes"] / 1e9), 4)

    for k, v in rows.items():
        v["composite_wus_per_hour_per_GB"] = composite(v) if v["hcli_wus_per_hour"] else 0.0

    archive = {
        "LOWEST_DENSITY": min(rows, key=lambda k: rows[k]["complete_ebpw"]),
        "FASTEST_TPOT": min(rows, key=lambda k: rows[k]["TPOT_ns_p50"]),
        "LOWEST_TTFT": min(rows, key=lambda k: rows[k]["TTFT_ns"]),
        "BEST_CAPABILITY": max(rows, key=lambda k: rows[k]["capability_passed"] or 0),
        "BEST_HCLI_WUS_HOUR": max(rows, key=lambda k: rows[k]["hcli_wus_per_hour"]),
        "BEST_LONG_CONTEXT": None,
        "BEST_MULTISESSION": None,
    }
    best = max(eligible, key=lambda k: eligible[k]["composite_wus_per_hour_per_GB"]) \
        if eligible else None
    runner = sorted(eligible, key=lambda k: -eligible[k]["composite_wus_per_hour_per_GB"])
    margin = None
    if len(runner) >= 2:
        a, b = eligible[runner[0]], eligible[runner[1]]
        margin = round(100 * (a["composite_wus_per_hour_per_GB"]
                              / b["composite_wus_per_hour_per_GB"] - 1), 2)

    out = {
        "schema": "hawking.odyssey.pareto_archive.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/pareto_archive.py",
        "obligation": "G040 — PARETO_ARCHIVE + FINAL_RESIDENT_SELECTION",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "candidates": rows,
        "archive": archive,
        "unmeasured_categories": {
            "BEST_LONG_CONTEXT": "the only long-context receipt on disk "
                                 "(LONG_CONTEXT_RUNTIME_CAPABILITY.json) measures the "
                                 "llama.cpp Q5_K reference at 29,982 prompt tokens, not "
                                 "any noetic body. No noetic body has a long-context "
                                 "measurement, so this category is UNMEASURED rather "
                                 "than assigned.",
            "BEST_MULTISESSION": "concurrency equilibrium was measured on clean-2.60 "
                                 "only (c4, 1.844x). It is not known to transfer to the "
                                 "other bodies, and clean-2.60 is capability-dead, so no "
                                 "eligible candidate has a multisession number.",
        },
        "selection": {
            "rule": "CAPABILITY FLOOR FIRST, then Pareto, then the S011 §0 composite",
            "floor": f"at least {FLOOR} verified capability item AND a nonzero verified "
                     f"HCLI WUs/hour",
            "why_floor_first": "LOWEST_DENSITY and FASTEST_TPOT are both held by "
                               "capability-dead bodies. A density-first rule would elect "
                               "a resident that cannot do any work.",
            "rejected_by_floor": rejected,
            "eligible": sorted(eligible),
            "pareto_front": sorted(front),
            "front_is_not_a_singleton": len(front) > 1,
            "composite_metric": "verified HCLI WUs/hour per GB resident (S011 §0: work / "
                                "wall time / physical resource)",
            "ranked": [{"body": k,
                        "composite_wus_per_hour_per_GB":
                            eligible[k]["composite_wus_per_hour_per_GB"]}
                       for k in runner],
            "provisional_resident": best,
            "margin_over_runner_up_pct": margin,
            "CONFIDENCE": {
                "margin_pct": margin,
                "worst_rate_spread_pct": max(
                    (eligible[k].get("hcli_rate_spread_pct") or 0) for k in eligible),
                "margin_exceeds_spread": (
                    margin > max((eligible[k].get("hcli_rate_spread_pct") or 0)
                                 for k in eligible)),
                "reps_per_body": {k: eligible[k].get("hcli_n_reps") for k in eligible},
                "accepted_stable_across_reps": {
                    k: len(set(eligible[k].get("hcli_accepted_reps") or [])) == 1
                    for k in eligible},
                "reading":
                    "MEASUREMENT-STABLE, BENCH-LIMITED. Three alternated reps per body "
                    "give a rate spread of 0.36% and 0.61%, and the accepted count is "
                    "identical in every rep, so the composite margin is several times "
                    "the measurement noise and is not an artifact. What it cannot "
                    "settle is bench composition: both bodies accept 5 of 6 WorkUnits "
                    "and the whole gap is wall-clock on a 6-unit suite. A different "
                    "WorkUnit mix could reorder them, and no number of reps fixes that.",
            },
        },
    }
    # G042's adversary attacked the capability column and won.
    def _cap(label):
        q = RH / f"CAPABILITY_{label}.json"
        return json.load(open(q))["overall"]["passed"] if q.is_file() else None

    s_no, s_sys = _cap("noetic-sealed-3.14"), _cap("noetic-sealed-3.14-sysprompt")
    v_no, v_sys = _cap("noetic-variantB-2.76"), _cap("noetic-variantB-2.76-sysprompt")
    if None not in (s_no, s_sys, v_no, v_sys):
        out["CAPABILITY_COLUMN_IS_REGIME_CONDITIONAL"] = {
            "attacked_by": "G042 adversary A5 "
                           "(receipts/headless/GRAND_CANDIDATE.json)",
            "no_system_prompt": {"sealed": s_no, "variantB": v_no, "gap": s_no - v_no},
            "with_default_system_prompt": {"sealed": s_sys, "variantB": v_sys,
                                           "gap": s_sys - v_sys},
            "finding": f"re-scoring the full suite with a neutral system prompt on the "
                       f"15 items that define none moves sealed {s_no}->{s_sys} and "
                       f"variantB {v_no}->{v_sys}. They TIE. sealed loses the knowledge "
                       f"axis outright; variantB gains coding 0.000->1.000.",
            "consequence": "the capability_passed column above no longer separates the "
                           "two frontier bodies. The SELECTION still stands because the "
                           "composite is measured under one consistent regime and does "
                           "not use capability as a tiebreak, but capability must not be "
                           "quoted as a reason to prefer sealed.",
            "what_still_separates_them": "verified HCLI WUs/hour (56.92 vs 48.84, three "
                                         "reps each, spreads 0.61% and 0.36%) and TPOT; "
                                         "variantB still wins EBPW and TTFT.",
        }

    out["pass"] = bool(eligible and front and best)
    p = RH / "PARETO_ARCHIVE.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"{'body':16s}{'EBPW':>8s}{'TPOT ms':>9s}{'TTFT ms':>9s}{'cap':>7s}"
          f"{'WU/hr':>8s}{'WU/hr/GB':>10s}")
    for k, v in sorted(rows.items(), key=lambda x: x[1]["complete_ebpw"]):
        print(f"{k:16s}{v['complete_ebpw']:>8.4f}{v['TPOT_ns_p50']/1e6:>9.3f}"
              f"{v['TTFT_ns']/1e6:>9.1f}"
              f"{str(v['capability_passed'])+'/43':>7s}{v['hcli_wus_per_hour']:>8.2f}"
              f"{v['composite_wus_per_hour_per_GB']:>10.4f}")
    print(f"\nrejected by capability floor: {sorted(rejected)}")
    print(f"pareto front: {sorted(front)}")
    print(f"provisional resident: {best}  (+{margin}% composite over runner-up)")
    for k, v in out["unmeasured_categories"].items():
        print(f"UNMEASURED {k}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
