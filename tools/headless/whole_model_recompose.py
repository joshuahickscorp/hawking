#!/usr/bin/env python3
"""N041 machinery (S025 §10-15): the whole-model information allocator + complete-EBPW closure.

The next number is not 2.25 (the MLP floor) -- it is the COMPLETE EBPW of the best
HETEROGENEOUS coherent Qwen executable, measured from the executable closure, not
inferred from one global BPW. Each organ carries its own coherent density floor;
the whole-model complete EBPW is the parameter-weighted sum:

    complete_ebpw = sum(organ_params * organ_ebpw) / total_params
                  = sum(organ_model_specific_bits) / total_params

This generator reads the per-organ parameter shares (NOETIC_ORGAN_CENSUS) and each
organ's best-known coherent floor (ORGAN_LIBRARY / ORGAN_DENSITY_FLOORS when N040
lands) and reports the allocation + the sub-3 verdict. Non-MLP floors are marked
PROVISIONAL until N040 measures them natively; the MLP 2.25 is CONFIRMED (4 ways).
"""
from __future__ import annotations
import json, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts" / "headless"
OUT = R / "WHOLE_MODEL_RECOMPOSE.json"


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()[:12]


def organ_params():
    """Per-organ parameter counts, derived from census byte shares at the uniform
    incumbent artifact density (bytes -> params via total ratio)."""
    c = load("NOETIC_ORGAN_CENSUS")
    art = c["artifact"]
    total_params = art["parameter_count"]
    total_bytes = art["bytes"]
    organs = c["organs"]
    out = {}
    for name, o in organs.items():
        b = o["physical"]["bytes"]
        out[name] = {
            "bytes": b,
            "params": round(b / total_bytes * total_params),
            "param_share": b / total_bytes,
        }
    return out, total_params


# Best-known COHERENT floor per organ. MLP is CONFIRMED (N021/N032/N033/N036/N038, 4 ways).
# The others are PROVISIONAL current-known values from ORGAN_FRONTIERS; N040 refines them.
KNOWN_FLOOR = {
    "mlp":            {"ebpw": 2.25,  "status": "CONFIRMED",   "src": "MLP density floor measured 4 ways (N039 QWEN_COMPLETION_RECEIPT)"},
    "deltanet":       {"ebpw": 4.125, "status": "PROVISIONAL", "src": "ORGAN_FRONTIERS deltanet floor; N040 tests recurrent transition program"},
    "attention_gqa":  {"ebpw": 4.25,  "status": "PROVISIONAL", "src": "ORGAN_FRONTIERS gqa floor; N040 tests head/KV redundancy"},
    "embedding":      {"ebpw": 4.125, "status": "PROVISIONAL", "src": "ORGAN_FRONTIERS embed floor; N040 tests table structure"},
    "output":         {"ebpw": 3.25,  "status": "PROVISIONAL", "src": "ORGAN_FRONTIERS lm_head floor; N040 tests table structure"},
}



def _measured_floors():
    """N040 ORGAN_DENSITY_FLOORS measured coherent floors, if present -- these override the
    provisional values (they reached complete_token composition on real activations)."""
    p = R / "ORGAN_DENSITY_FLOORS.json"
    if not p.is_file():
        return {}
    d = json.loads(p.read_text())
    organs = d.get("organs") or d
    def floor(o):
        if isinstance(o, dict):
            for k in ("coherent_floor_ebpw", "floor_ebpw", "lowest_coherent_ebpw", "complete_ebpw", "ebpw"):
                if k in o and isinstance(o[k], (int, float)):
                    return o[k]
            for v in o.values():
                r = floor(v)
                if r is not None:
                    return r
        return None
    out = {}
    m = {"deltanet": "deltanet", "gqa_attention": "attention_gqa",
         "embedding_output": ["embedding", "output"]}
    for src, dst in m.items():
        o = organs.get(src) if isinstance(organs, dict) else None
        fl = floor(o) if o else None
        if fl is None:
            continue
        for key in ([dst] if isinstance(dst, str) else dst):
            out[key] = fl
    return out


def recompose():
    op, total_params = organ_params()
    measured = _measured_floors()
    # map census organ names -> floor keys
    alias = {"mlp": "mlp", "deltanet": "deltanet", "attention_gqa": "attention_gqa",
             "embedding": "embedding", "output": "output"}
    alloc = []
    total_bits = 0.0
    for cname, floor_key in alias.items():
        if cname not in op:
            continue
        f = dict(KNOWN_FLOOR[floor_key])
        if floor_key in measured:
            f = {"ebpw": measured[floor_key], "status": "MEASURED",
                 "src": "N040 ORGAN_DENSITY_FLOORS.json (grouped_absmax q3, complete_token on real X)"}
        params = op[cname]["params"]
        bits = params * f["ebpw"]
        total_bits += bits
        alloc.append({
            "organ": cname, "params": params, "param_share": round(op[cname]["param_share"], 6),
            "ebpw": f["ebpw"], "floor_status": f["status"], "source": f["src"],
            "organ_bits": bits, "contribution_to_complete_ebpw": round(bits / total_params, 6),
        })
    complete_ebpw = total_bits / total_params
    # ACTIVE EBPW/token: census active_bytes_per_token is at the uniform incumbent 4.2527 bpw;
    # scale each organ by (chosen_ebpw / 4.2527), sum active bits, divide by params.
    INCUMBENT_BPW = 4.2527
    census = load("NOETIC_ORGAN_CENSUS")["organs"]
    active_bits = 0.0
    for a in alloc:
        o = census.get(a["organ"])
        if not o:
            continue
        abpt = o["physical"]["active_bytes_per_token"]
        active_bits += abpt * 8 * (a["ebpw"] / INCUMBENT_BPW)
        a["active_bytes_per_token_at_floor"] = round(abpt * (a["ebpw"] / INCUMBENT_BPW))
    active_ebpw_per_token = active_bits / total_params
    return alloc, complete_ebpw, total_params, active_ebpw_per_token


def main():
    alloc, complete_ebpw, total_params, active_ebpw_per_token = recompose()
    provisional = [a["organ"] for a in alloc if a["floor_status"] == "PROVISIONAL"]
    doc = {
        "schema": "hawking.headless.whole_model_recompose.v1",
        "obligation": "N041 (S025 §10-15) -- BASELINE with current-known floors; N040 refines non-MLP",
        "git_head": git_head(),
        "generated_by": "tools/headless/whole_model_recompose.py",
        "hand_authored": False,
        "method": "complete_ebpw = sum(organ_params * organ_coherent_floor_ebpw) / total_params "
                  "= sum(organ_model_specific_bits) / total_params (executable-closure sum, not one global BPW)",
        "parent_parameter_count": total_params,
        "allocation": alloc,
        "current_qwen_complete_ebpw": round(complete_ebpw, 6),
        "active_ebpw_per_token": round(active_ebpw_per_token, 6),
        "historical_complete_ebpw": 3.1393,
        "below_3_0": complete_ebpw < 3.0,
        "headline": (
            f"With the MLP at its CONFIRMED 2.25 floor and the non-MLP organs at their current "
            f"PROVISIONAL floors, the whole-model complete EBPW is {complete_ebpw:.4f} -- "
            f"{'BELOW' if complete_ebpw < 3.0 else 'ABOVE'} 3.0. The MLP result alone "
            f"({'already ' if complete_ebpw < 3.0 else ''}moved it from the historical 3.1393)."
        ),
        "provisional_organs": provisional,
        "refined_by": "N040 ORGAN_DENSITY_FLOORS -- when it lands, re-run to replace the provisional "
                      "non-MLP floors with measured coherent floors; DeltaNet (20.7% of params) is the "
                      "largest non-MLP lever.",
        "note": "This is the allocator + closure machinery (N041) run on best-known data. It is NOT the "
                "final heterogeneous executable: native execution (QWEN_ZERO_PARENT_RUNTIME_DEPENDENCY) "
                "and the reprofiled model-reachable roof (S025 §16,§17) are separate and still owed.",
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"complete EBPW baseline: {complete_ebpw:.4f}  below_3.0={complete_ebpw < 3.0}")
    for a in alloc:
        print(f"  {a['organ']:14} share={a['param_share']:.3f} ebpw={a['ebpw']:<5} "
              f"contrib={a['contribution_to_complete_ebpw']:.4f} [{a['floor_status']}]")
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
