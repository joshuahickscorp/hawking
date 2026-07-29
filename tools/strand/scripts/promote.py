#!/usr/local/bin/python3
"""promote.py - the shared promotion grammar for the STRAND quality-density frontier.

One small tool that every result passes through, so the orchestration speaks gates
instead of vibes. It does four things the roadmap asks for:

  1. computes loss tax automatically when a bf16 anchor is known
       loss_tax_nats = ln(PPL_quant / PPL_bf16)
  2. refuses to promote a result that lacks billed evidence (ppl, bpw, harness id)
  3. applies the docs' explicit kill bars per gate type
  4. stamps the decision back INTO the json, so a result self-describes its state

It reads any STRAND result json (eval build_record, strand_debias_ppl_ab_v1,
pv lineage row, or a frontier leg with {ppl, eff_bpw}), prints ONE grammar line,
and (unless --dry-run) writes a `promotion` block into the json.

Grammar states (small on purpose):
  INCOMPLETE     - missing ppl / bpw / harness identity; cannot be trusted to scale
  KILLED         - failed its gate's kill bar
  LOCAL_PASS     - passed a local gate but not yet earned cloud scale
  PROMOTE_CLOUD  - passed; a pod confirmation run is justified
  GATED          - measured and billed, but no decisive bar applies (informational)

The pod chain must refuse to scale anything whose promotion.state != PROMOTE_CLOUD.
"""

import argparse
import json
import math
import os
import sys

# bf16 truth anchors (held-out WikiText-2 PPL, same harness). Matched by substring
# of the model path/basename. These are the denominators of the loss-tax metric.
BF16_ANCHORS = {
    "qwen-05b": 12.536, "qwen2.5-0.5b": 12.536, "qwen2_5_0_5b": 12.536,
    "qwen-7b": 6.629, "qwen2.5-7b": 6.629,
    "qwen-14b": 5.102, "qwen-32b": 4.778,
    "llama2-7b": 5.535, "llama-2-7b": 5.535,
}

# decisive bars lifted from docs/STRAND-quality-density-frontier.md
PV_PROMOTE_PPL = 30.0      # §4.A: <=30 -> promote dp-d4-r2 PV to main 2-bit lane
PV_WEAK_PPL = 36.0         # §4.A: 30-36 alive-but-weak; >=36 do not scale
PRIOR_PV_FLOOR = 26.77     # best prior 0.5B PV floor; beating it is the real win
DEBIAS_ADOPT = 0.995       # §4.C: adopt if PPL_B <= PPL_A * 0.995
SEED_ADOPT = 0.995         # §4.E: adopt seed/basis if >=0.5% held-out PPL gain


def anchor_for(model_path):
    if not model_path:
        return None, None
    key = os.path.basename(os.path.normpath(str(model_path))).lower()
    full = str(model_path).lower()
    for k, v in BF16_ANCHORS.items():
        if k in key or k in full:
            return v, k
    return None, None


def loss_tax(ppl, bf16):
    if not ppl or not bf16 or ppl <= 0 or bf16 <= 0:
        return None
    return round(math.log(ppl / bf16), 4)


def has_harness_id(rec):
    """A result may be scaled only if its eval is identifiable and reproducible."""
    if not isinstance(rec, dict):
        return False
    if rec.get("harness_key8") or rec.get("harness_key"):
        return True
    # fall back to the minimal identity tuple
    need = ("dataset_fp", "ctx", "chunks")
    return all(rec.get(k) is not None for k in need)


def detect(obj):
    """Return (gate_type, primary_record) for the json shape."""
    s = obj.get("schema")
    if s == "strand_debias_ppl_ab_v1":
        return "debias_ab", obj
    if s == "strand_actmean_v1":
        return "actmean", obj
    if "ppl_after" in obj or ("ppl_before" in obj and "steps" in obj):
        return "pv", obj
    # eval build_record or frontier leg: has a ppl somewhere
    if "ppl" in obj:
        return "eval", obj
    if "baseline" in obj and isinstance(obj.get("baseline"), dict):
        return "debias_ab", obj
    return "unknown", obj


def gate_debias(obj):
    ratio = obj.get("ratio")
    contaminated = obj.get("contamination_warning")
    base = obj.get("baseline", {})
    bpw = base.get("eff_bpw")
    reasons = []
    if contaminated or (ratio is not None and abs(ratio - 1.0) < 1e-9):
        return "KILLED", [f"contamination: identical A/B PPL (ratio={ratio})"], None, bpw
    if ratio is None:
        return "INCOMPLETE", ["no ratio field"], None, bpw
    pct = (ratio - 1.0) * 100.0
    deb_ppl = obj.get("debiased", {}).get("ppl")
    if ratio <= DEBIAS_ADOPT:
        reasons.append(f"ADOPT: debiased PPL {pct:+.3f}% <= -0.5% bar")
        state = "LOCAL_PASS"
    elif ratio >= 1.0:
        reasons.append(f"KILL: debias did not help ({pct:+.3f}%)")
        state = "KILLED"
    else:
        reasons.append(f"INCONCLUSIVE: {pct:+.3f}% (between -0.5% adopt and 0% kill)")
        state = "GATED"
    # de-bias is a local correction gate; it never auto-promotes to cloud by itself
    return state, reasons, deb_ppl, bpw


def gate_pv(obj):
    after = obj.get("ppl_after")
    before = obj.get("ppl_before")
    reasons = []
    if after is None:
        return "INCOMPLETE", ["no ppl_after"], None
    if before:
        reasons.append(f"PV moved {before:.2f} -> {after:.2f} ({(after/before-1)*100:+.1f}%)")
    reasons.append(f"vs prior PV floor {PRIOR_PV_FLOOR}: {after-PRIOR_PV_FLOOR:+.2f}")
    if after <= PV_PROMOTE_PPL and after < PRIOR_PV_FLOOR:
        return "PROMOTE_CLOUD", reasons + ["beats floor and <=30: promote to main 2-bit lane, run WSD A/B"], after
    if after <= PV_PROMOTE_PPL:
        return "LOCAL_PASS", reasons + ["<=30 but not below prior floor: WSD/progressive before scale"], after
    if after <= PV_WEAK_PPL:
        return "LOCAL_PASS", reasons + ["30-36 alive-but-weak: try WSD/progressive, do not scale yet"], after
    return "KILLED", reasons + [">=36: direct cosine PV insufficient; pivot to progressive/de-bias/basis"], after


def gate_eval(obj):
    """Generic eval / frontier leg: bill completeness, compute tax, stay informational."""
    ppl = obj.get("ppl")
    bpw = obj.get("eff_bpw")
    reasons = []
    if ppl is None:
        return "INCOMPLETE", ["no ppl"], None, bpw
    if bpw is None:
        reasons.append("eff_bpw missing: cannot bill density")
    if not has_harness_id(obj):
        reasons.append("no harness identity: not reproducible enough to scale")
    state = "GATED" if (bpw is not None and has_harness_id(obj)) else "INCOMPLETE"
    reasons.append("informational: needs an explicit target/bar to promote")
    return state, reasons, bpw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", help="path to a STRAND result json")
    ap.add_argument("--dry-run", action="store_true", help="print verdict, do not write back")
    ap.add_argument("--quiet", action="store_true", help="only the grammar line")
    ap.add_argument("--model", default=None,
                    help="model hint for bf16 anchor when the json stores only a recon dir")
    ap.add_argument("--bf16", type=float, default=None,
                    help="explicit bf16 PPL anchor (overrides auto-detection)")
    args = ap.parse_args()

    with open(args.result) as f:
        obj = json.load(f)

    gate_type, rec = detect(obj)
    bpw = None
    primary_ppl = None
    state, reasons = "GATED", []

    if gate_type == "debias_ab":
        state, reasons, primary_ppl, bpw = gate_debias(obj)
        model_path = obj.get("baseline", {}).get("model") or obj.get("baseline", {}).get("model_path")
    elif gate_type == "pv":
        state, reasons, primary_ppl = gate_pv(obj)
        bpw = obj.get("eff_bpw") or obj.get("bpw")
        model_path = obj.get("model") or obj.get("model_path") or "qwen-05b"
    elif gate_type == "eval":
        state, reasons, bpw = gate_eval(obj)
        primary_ppl = obj.get("ppl")
        model_path = obj.get("model") or obj.get("model_path")
    elif gate_type == "actmean":
        state, reasons = "GATED", ["calibration artifact: feeds de-bias gate, not promotable"]
        model_path = obj.get("model")
    else:
        state, reasons = "INCOMPLETE", ["unrecognized json shape"]
        model_path = obj.get("model") or obj.get("model_path")

    # anchor resolution order: --bf16 override, --model hint, stamped source_model,
    # then the json's own model path. Keeps loss-tax auto-computable for callers
    # (conductor/pod-chain) that know the model even when the eval stored a recon dir.
    source_hint = args.model or obj.get("source_model") or model_path
    if args.bf16 is not None:
        bf16, anchor_key = args.bf16, (args.model or "override")
    else:
        bf16, anchor_key = anchor_for(source_hint)
    tax = loss_tax(primary_ppl, bf16)

    promotion = {
        "state": state,
        "gate_type": gate_type,
        "loss_tax_nats": tax,
        "bf16_anchor": bf16,
        "anchor_key": anchor_key,
        "eff_bpw": bpw,
        "primary_ppl": primary_ppl,
        "reasons": reasons,
        "billed_complete": state != "INCOMPLETE",
    }
    obj["promotion"] = promotion

    if not args.dry_run:
        with open(args.result, "w") as f:
            json.dump(obj, f, indent=2)

    taxs = f"tax={tax}" if tax is not None else "tax=?"
    bpws = f"bpw={bpw}" if bpw is not None else "bpw=?"
    ppls = f"ppl={primary_ppl}" if primary_ppl is not None else "ppl=?"
    line = f"PROMOTE state={state} {gate_type} {ppls} {bpws} {taxs} :: {'; '.join(reasons)}"
    print(line)
    if not args.quiet and not args.dry_run:
        print(f"[promote] stamped {args.result}", file=sys.stderr)
    # exit code encodes the decision for shell gates: 0 promote, 10 local, 20 gated, 30 kill, 40 incomplete
    return {"PROMOTE_CLOUD": 0, "LOCAL_PASS": 10, "GATED": 20,
            "KILLED": 30, "INCOMPLETE": 40}.get(state, 20)


if __name__ == "__main__":
    sys.exit(main())
