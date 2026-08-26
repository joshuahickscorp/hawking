#!/usr/bin/env python3
"""Does the transferred METHOD hold on a new architecture family?

The law under test, currently sealed at FAMILY_TRANSFERRED because both measurements were
Qwen: at matched bits per weight, a least-squares-refit affine codec beats generic
grouped-absmax round-to-nearest.

Promotion to ARCHITECTURE_GENERAL needs two DISTINCT architecture families. This runs the
identical evaluator from tools/odyssey/cold_vs_transfer.py on any dense MLP, so the only
thing that changes between runs is the specimen.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/odyssey"))
from cold_vs_transfer import CANDIDATES, swiglu_out, rel_fro  # noqa: E402

MLP_NAMES = [("mlp", "gate_proj", "up_proj", "down_proj"),
             ("feed_forward", "gate_proj", "up_proj", "down_proj"),
             ("feed_forward", "w1", "w3", "w2")]


def load_mlp(md, idx, layer):
    """Resolve the MLP by ROLE, the same way the recognizer does."""
    import torch
    from safetensors import safe_open

    def get(name):
        with safe_open(md / idx[name], framework="pt") as f:
            return f.get_tensor(name).float().numpy().astype(np.float64)

    for block, g, u, d in MLP_NAMES:
        pre = f"model.layers.{layer}.{block}."
        if pre + g + ".weight" in idx:
            return (get(pre + g + ".weight"), get(pre + u + ".weight"),
                    get(pre + d + ".weight")), f"{block}.{{{g},{u},{d}}}"
    raise SystemExit(f"no dense MLP at layer {layer}; tried {MLP_NAMES}")


def weight_space(name, g, u, d):
    """Rank the same candidate by weight-space error, the method the campaign REFUTED.

    Included so the two rankings can be compared on THIS architecture rather than assumed
    to disagree the way they did on Qwen.
    """
    fn = CANDIDATES[name]
    errs, sizes = [], []
    for W in (g, u, d):
        Wq, _ = fn(W)
        errs.append(rel_fro(Wq, W))
        sizes.append(W.size)
    return float(np.average(errs, weights=sizes))


def evaluate(name, g, u, d, Xhold):
    fn = CANDIDATES[name]
    gq, bg = fn(g)
    uq, bu = fn(u)
    dq, bd = fn(d)
    bpw = float(np.average([bg, bu, bd], weights=[g.size, u.size, d.size]))
    # the identical SwiGLU used by the specimen-#2 experiment, imported rather than
    # reimplemented, so the two runs cannot drift apart
    ref = swiglu_out(Xhold, g, u, d)
    got = swiglu_out(Xhold, gq, uq, dq)
    return {"candidate": name, "bpw": round(bpw, 4),
            "held_out_rel_fro": round(rel_fro(got, ref), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--label", required=True)
    ap.add_argument("--architecture-family", required=True)
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    md = Path(a.model_dir)
    cap = json.load(open(Path(a.capture) / "CAPTURE.json"))
    if not cap.get("all_layers_finite"):
        raise SystemExit("capture is not certified finite; refusing to rank on it")
    idx = json.load(open(md / "model.safetensors.index.json"))["weight_map"]
    X = np.load(Path(a.capture) / f"X_layer{a.layer}.npy").astype(np.float64)
    Xhold = X[X.shape[0] // 2:]
    (g, u, d), resolved = load_mlp(md, idx, a.layer)

    t0 = time.time()
    scores = {n: evaluate(n, g, u, d, Xhold) for n in CANDIDATES}
    for n, v in scores.items():
        v["weight_space_rel_fro"] = round(weight_space(n, g, u, d), 6)

    tiers = {}
    for v in scores.values():
        tiers.setdefault(round(v["bpw"], 3), []).append(v)
    matched = []
    for bpw, vs in sorted(tiers.items()):
        fam = {}
        for v in sorted(vs, key=lambda v: v["held_out_rel_fro"]):
            fam.setdefault("affine" if v["candidate"].startswith("affine") else "generic", v)
        if "affine" in fam and "generic" in fam:
            aa, gg = fam["affine"], fam["generic"]
            matched.append({"bpw": bpw,
                            "seeded_family_candidate": aa["candidate"],
                            "seeded_family_rel_fro": aa["held_out_rel_fro"],
                            "generic_candidate": gg["candidate"],
                            "generic_rel_fro": gg["held_out_rel_fro"],
                            "seeded_is_better": aa["held_out_rel_fro"] < gg["held_out_rel_fro"],
                            "error_ratio_generic_over_seeded": round(
                                gg["held_out_rel_fro"] / aa["held_out_rel_fro"], 3)})
    wins = sum(1 for m in matched if m["seeded_is_better"])

    # Does weight-space error rank these the same way held-out activations do?
    by_act = [k for k, v in sorted(scores.items(), key=lambda kv: kv[1]["held_out_rel_fro"])]
    by_wt = [k for k, v in sorted(scores.items(), key=lambda kv: kv[1]["weight_space_rel_fro"])]
    disagreements = [{"candidate": c, "rank_by_activations": by_act.index(c),
                      "rank_by_weight_space": by_wt.index(c)}
                     for c in scores if by_act.index(c) != by_wt.index(c)]
    # the decisive case: does weight-space pick a different WINNER at a matched bit tier?
    tier_flips = []
    for m in matched:
        a_name, g_name = m["seeded_family_candidate"], m["generic_candidate"]
        wt_winner = min((a_name, g_name), key=lambda c: scores[c]["weight_space_rel_fro"])
        act_winner = min((a_name, g_name), key=lambda c: scores[c]["held_out_rel_fro"])
        if wt_winner != act_winner:
            tier_flips.append({"bpw": m["bpw"], "weight_space_picks": wt_winner,
                               "activations_pick": act_winner})

    out = {
        "schema": "hawking.headless.matched_bits_probe.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/matched_bits_probe.py",
        "obligation": "G028 extension — can LAW-FITTED-AFFINE-BEATS-RTN reach "
                      "ARCHITECTURE_GENERAL (directive §91)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "specimen": {"label": a.label, "model_dir": str(md),
                     "architecture_family": a.architecture_family,
                     "model_type": cap.get("model_type"),
                     "mlp_resolved_as": resolved},
        "activations": {"real_not_synthetic": True, "n_tokens": int(X.shape[0]),
                        "n_held_out": int(Xhold.shape[0]),
                        "exact_because": cap["exact_because"],
                        "all_layers_finite": True},
        "layer": a.layer,
        "full_grid": scores,
        "matched_bits_tiers": matched,
        "n_tiers": len(matched), "n_tiers_seeded_wins": wins,
        "law_holds_here": wins == len(matched) and len(matched) >= 3,
        "weight_space_vs_activations": {
            "question": "does weight-space error rank candidates the same way held-out real "
                        "activations do, on THIS architecture?",
            "ranking_by_activations": by_act,
            "ranking_by_weight_space": by_wt,
            "n_candidates_ranked_differently": len(disagreements),
            "disagreements": disagreements,
            "matched_tier_winner_flips": tier_flips,
            "orderings_identical": not disagreements,
            "why_it_matters": "TR-METHOD-HELDOUT-ACTIVATIONS is sealed at FAMILY_TRANSFERRED "
                              "because both prior measurements were Qwen. A disagreement "
                              "here is a second architecture family for it.",
        },
        "wall_s": round(time.time() - t0, 1),
        "pass": bool(matched),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"{a.label} ({a.architecture_family}) layer {a.layer}, MLP resolved as {resolved}")
    for m in matched:
        print(f"  {m['bpw']:>6} bpw   affine {m['seeded_family_rel_fro']:.6f}   "
              f"generic {m['generic_rel_fro']:.6f}   ratio {m['error_ratio_generic_over_seeded']}"
              f"   {'affine wins' if m['seeded_is_better'] else 'GENERIC WINS'}")
    print(f"  law holds here: {out['law_holds_here']} ({wins}/{len(matched)} tiers)")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
