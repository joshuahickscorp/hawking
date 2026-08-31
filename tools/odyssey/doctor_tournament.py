#!/usr/bin/env python3
"""G035 — DOCTOR_GRAND_TOURNAMENT (S011 §7, §8, §9, §16, §17).

The 39-technique library run as ACTIVE machinery: diagnose each organ, rank techniques
by whether their PRECONDITION actually holds on this model, and expand only survivors.
S011 §8 forbids a blind q4/q3/q2 sweep, so nothing here is ranked by bit width.

Every technique family has a precondition that is cheap to test on the weights alone:

  COORDINATES  rotation only helps if there are outliers to redistribute
  FACTOR       low-rank only helps if the singular spectrum actually decays
  SHARE        a shared basis only helps if layers are correlated
  ELIMINATE    structural pruning only helps if some structure is near-zero

Those are measured here on real parent weights. Techniques whose precondition fails are
closed cheaply, which is the whole point of §9: cheap discriminating tests first,
expansion only for survivors.

Several stages are already adjudicated by measurements this campaign made -- the
tokenizer elimination (G036), the decoding census (G038), state (G037), the quantize
floor (G034) -- and those verdicts are cited rather than re-run.
"""
import json, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
PARENT = Path("/Volumes/corpdrive/personalmodel/correspondent/qwen3.8-27b-abliterated-bf16")

CANONICAL_ORDER = ["ELIMINATE", "COORDINATES", "SHARE", "FACTOR", "GENERATE", "CODEBOOK",
                   "ROUTE", "SENSITIVITY", "HEAL", "QUANTIZE", "NATIVE", "STATE",
                   "CONDITIONAL", "DECODING"]

# library family -> canonical stage
FAMILY_STAGE = {
    "DOC-COORDINATES": "COORDINATES",
    "learned/function-preserving rotations + incoherence processing": "COORDINATES",
    "DOC-HEALING": "HEAL",
    "DOC-REPRESENTATION": "QUANTIZE",
    "ultra-low-bit PTQ": "QUANTIZE",
    "additive/vector codebooks": "CODEBOOK",
    "shared_basis / cross-layer coefficients": "SHARE",
    "DOC-STATE": "STATE",
    "KV & state compression": "STATE",
    "linear-attention / SSM / DeltaNet / gated-delta compression": "STATE",
    "DOC-CONDITIONAL": "CONDITIONAL",
    "activation sparsity + conditional compute": "CONDITIONAL",
    "DOC-DECODE": "DECODING",
    "speculative / multi-token / self-speculative decoding": "DECODING",
    "tokenizer / vocabulary reduction": "ELIMINATE",
}

# Layer choice matters: full_attention_interval=4 puts self_attn on layers 3, 7, 11...
# and linear_attn on every other layer. Probing 3/31/59 for BOTH sampled only
# full-attention layers, so deltanet silently returned zero tensors and vanished from
# the diagnosis. Each organ now names the layers it actually lives on.
PROBE_TENSORS = [
    ("mlp", "model.language_model.layers.{L}.mlp.gate_proj.weight", (2, 30, 58)),
    ("mlp", "model.language_model.layers.{L}.mlp.down_proj.weight", (2, 30, 58)),
    ("attention_gqa", "model.language_model.layers.{L}.self_attn.q_proj.weight",
     (3, 31, 59)),
    ("deltanet", "model.language_model.layers.{L}.linear_attn.out_proj.weight",
     (2, 30, 58)),
]


def probes(max_rows=2048):
    """Cheap, CPU-only, weights-only preconditions."""
    import torch
    from safetensors import safe_open
    idx = json.load(open(PARENT / "model.safetensors.index.json"))["weight_map"]
    out, by_organ_layers, missing = {}, {}, []

    for organ, pat, layers in PROBE_TENSORS:
        for L in layers:
            name = pat.format(L=L)
            if name not in idx:
                missing.append(name)
                continue
            with safe_open(str(PARENT / idx[name]), framework="pt") as f:
                W = f.get_tensor(name).to(torch.float32)
            if W.shape[0] > max_rows:
                W = W[:max_rows]
            flat = W.flatten()
            std = flat.std().item()
            # COORDINATES: excess kurtosis and the tail mass rotation would redistribute
            z = (flat - flat.mean()) / (std + 1e-12)
            kurt = (z.pow(4).mean() - 3.0).item()
            outlier_frac = (z.abs() > 4).float().mean().item()
            # FACTOR: does the spectrum decay?
            sv = torch.linalg.svdvals(W)
            energy = (sv ** 2)
            cum = energy.cumsum(0) / energy.sum()
            r50 = int((cum < 0.50).sum()) + 1
            r90 = int((cum < 0.90).sum()) + 1
            n = min(W.shape)
            # ELIMINATE: near-zero output rows
            rown = W.norm(dim=1)
            dead_rows = (rown < 0.01 * rown.mean()).float().mean().item()
            out.setdefault(organ, []).append({
                "tensor": name, "shape": list(W.shape),
                "excess_kurtosis": round(kurt, 3),
                "outlier_frac_beyond_4sigma": round(outlier_frac, 6),
                "rank_for_50pct_energy": r50, "rank_for_90pct_energy": r90,
                "full_rank": n,
                "r90_over_full_rank": round(r90 / n, 4),
                "near_zero_row_frac": round(dead_rows, 6),
            })
            by_organ_layers.setdefault(organ, {})[L] = W[:512, :512].clone()

    # SHARE: are the same tensors correlated ACROSS layers?
    share = {}
    for organ, mats in by_organ_layers.items():
        Ls = sorted(mats)
        cos = []
        for i in range(len(Ls)):
            for j in range(i + 1, len(Ls)):
                a, b = mats[Ls[i]].flatten(), mats[Ls[j]].flatten()
                c = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                cos.append({"layers": [Ls[i], Ls[j]], "cosine": round(c, 5)})
        share[organ] = {"pairs": cos,
                        "max_abs_cosine": round(max(abs(p["cosine"]) for p in cos), 5)}
    return out, share, missing


def main():
    lib = json.load(open(RH / "DOCTOR_TECHNIQUE_LIBRARY.json"))["techniques"]
    stats, share, missing = probes()

    def agg(organ, key):
        v = [r[key] for r in stats.get(organ, [])]
        return round(sum(v) / len(v), 6) if v else None

    diagnosis = {}
    for organ in stats:
        diagnosis[organ] = {
            "measurements": stats[organ],
            "mean_excess_kurtosis": agg(organ, "excess_kurtosis"),
            "mean_outlier_frac": agg(organ, "outlier_frac_beyond_4sigma"),
            "mean_r90_over_full_rank": agg(organ, "r90_over_full_rank"),
            "mean_near_zero_row_frac": agg(organ, "near_zero_row_frac"),
            "max_cross_layer_cosine": share.get(organ, {}).get("max_abs_cosine"),
        }

    # Thresholds are judgment calls, stated so they can be argued with. The COORDINATES
    # verdict is the sensitive one: it flips at an excess-kurtosis threshold of about
    # 1.36, which is exactly deltanet's value.
    THRESHOLDS = {
        "COORDINATES_excess_kurtosis": 1.0,
        "COORDINATES_outlier_frac": 0.001,
        "FACTOR_r90_over_full_rank": 0.5,
        "SHARE_max_abs_cosine": 0.3,
        "ELIMINATE_near_zero_row_frac": 0.01,
    }

    # per-organ, because a single max() over organs hides which organ carries the signal
    per_organ_coordinates = {
        o: {"excess_kurtosis": d["mean_excess_kurtosis"],
            "outlier_frac": d["mean_outlier_frac"],
            "holds": (d["mean_excess_kurtosis"] > THRESHOLDS["COORDINATES_excess_kurtosis"]
                      or d["mean_outlier_frac"] > THRESHOLDS["COORDINATES_outlier_frac"])}
        for o, d in diagnosis.items()}

    # PRECONDITIONS, decided from the measurements above
    def precond(stage):
        if stage == "COORDINATES":
            worst = max(d["mean_outlier_frac"] for d in diagnosis.values())
            k = max(d["mean_excess_kurtosis"] for d in diagnosis.values())
            holds = worst > 0.001 or k > 1.0
            return holds, (f"max outlier fraction beyond 4 sigma = {worst}, max excess "
                           f"kurtosis = {k}. Rotation redistributes outliers; with a "
                           f"{'heavy' if holds else 'near-Gaussian'} tail there is "
                           f"{'something' if holds else 'nothing'} to redistribute.")
        if stage == "FACTOR":
            best = min(d["mean_r90_over_full_rank"] for d in diagnosis.values())
            holds = best < 0.5
            return holds, (f"the best organ needs {best:.1%} of full rank to hold 90% of "
                           f"spectral energy. Low-rank pays only if that is well below "
                           f"100%.")
        if stage == "SHARE":
            best = max((d["max_cross_layer_cosine"] or 0) for d in diagnosis.values())
            holds = best > 0.3
            return holds, (f"max |cosine| between the same tensor in different layers = "
                           f"{best}. A shared basis needs layers to actually resemble "
                           f"each other.")
        if stage == "ELIMINATE":
            best = max(d["mean_near_zero_row_frac"] for d in diagnosis.values())
            holds = best > 0.01
            return holds, (f"largest near-zero output-row fraction = {best}. Structural "
                           f"pruning needs structure that is already doing nothing.")
        return None, "no weights-only precondition; adjudicated by campaign evidence"

    ADJUDICATED = {
        "ELIMINATE": ("receipts/headless/TOKENIZER_GRAVITY.json",
                      "vocabulary elimination measured and REFUSED: 10.98% payload saved "
                      "costs 2.205x mean token inflation on held-out language"),
        "QUANTIZE": ("receipts/headless/VARIANT_LOCALIZATION.json",
                     "the MLP per-group bias is load-bearing (variantA 0/43 without it); "
                     "q2f 2.25 bpw is the floor under the tested conditions"),
        "STATE": ("receipts/headless/STATE_GRAVITY.json",
                  "KV precision ranked LAST: only 16 of 64 layers hold KV and the flat "
                  "72 MiB recurrent state dominates below 1,152 tokens. Prefill, not "
                  "state precision, is the lever"),
        "DECODING": ("receipts/headless/DECODING_GRAVITY.json",
                     "MTP machinery found on disk (1.579% of the model, never packed); "
                     "full-size drafts measured at 0.50-0.75 acceptance but 0.741-0.876x "
                     "speedup because draft cost equals verify cost"),
    }

    stages = {}
    for st in CANONICAL_ORDER:
        techs = [t for t in lib if FAMILY_STAGE.get(t["family"]) == st]
        holds, why = precond(st)
        adj = ADJUDICATED.get(st)
        if adj:
            verdict = "ADJUDICATED_BY_CAMPAIGN_MEASUREMENT"
        elif not techs:
            # measuring a precondition for a stage the library cannot act on is not a
            # refutation of anything; say the library has no technique there
            verdict = "NO_TECHNIQUE_IN_LIBRARY"
        elif holds is None:
            verdict = "UNTESTED_NEEDS_EXPANSION"
        elif holds:
            verdict = "PROBE_SUPPORTS_EXPAND"
        else:
            verdict = "PROBE_REFUTES_CLOSE_CHEAPLY"
        stages[st] = {
            "n_techniques": len(techs),
            "techniques": [t["id"] for t in techs],
            "precondition_holds": holds,
            "precondition_evidence": why,
            "campaign_verdict": adj[1] if adj else None,
            "campaign_receipt": adj[0] if adj else None,
            "verdict": verdict,
        }

    routed = sum(s["n_techniques"] for s in stages.values())
    survivors = [s for s, v in stages.items() if v["verdict"] == "PROBE_SUPPORTS_EXPAND"]
    closed = [s for s, v in stages.items() if v["verdict"] == "PROBE_REFUTES_CLOSE_CHEAPLY"]
    frontier = [s for s, v in stages.items()
                if v["verdict"] == "UNTESTED_NEEDS_EXPANSION" and v["n_techniques"]]
    n_closed_tech = sum(stages[s]["n_techniques"] for s in closed)
    n_frontier_tech = sum(stages[s]["n_techniques"] for s in frontier)

    out = {
        "schema": "hawking.odyssey.doctor_tournament.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/doctor_tournament.py",
        "obligation": "G035 — DOCTOR_GRAND_TOURNAMENT",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "canonical_order": CANONICAL_ORDER,
        "library_size": len(lib),
        "techniques_routed_to_a_stage": routed,
        "unrouted": [t["id"] for t in lib if FAMILY_STAGE.get(t["family"]) is None],
        "diagnosis": diagnosis,
        "probe_method": {
            "what": "weights-only preconditions on real parent tensors, 3 layers "
                    "(early/mid/late) across mlp, attention and deltanet",
            "why": "S011 §9 asks for cheap discriminating tests before expansion. A "
                   "technique whose precondition fails on the weights cannot be rescued "
                   "by tuning, so it closes without a GPU run.",
            "cost": "CPU only, no GPU, no model load into the runtime",
            "limit": "a precondition failing is strong evidence against a family, not a "
                     "proof. It bounds where search is worth spending, which is what a "
                     "tournament is for.",
        },
        "thresholds": THRESHOLDS,
        "threshold_honesty": "these cut-offs are judgment calls, not derived constants. "
                             "The COORDINATES verdict is the sensitive one: it rests "
                             "entirely on deltanet's excess kurtosis of "
                             f"{diagnosis.get('deltanet', {}).get('mean_excess_kurtosis')} "
                             "against a threshold of 1.0, and would flip if the threshold "
                             "moved to 1.4.",
        "coordinates_per_organ": per_organ_coordinates,
        "coordinates_s011_16_routing": {
            "prior": "Qwen's earlier whole-model rotation result was NEGATIVE, and S011 "
                     "§16 forbids relaunching it blindly -- only changed-condition "
                     "probes are allowed.",
            "changed_condition_found": "the tail structure is NOT uniform across organs. "
                                       "deltanet carries excess kurtosis "
                                       f"{diagnosis.get('deltanet', {}).get('mean_excess_kurtosis')} "
                                       "against "
                                       f"{diagnosis.get('mlp', {}).get('mean_excess_kurtosis')} "
                                       "for mlp -- roughly 6x. A whole-model rotation "
                                       "test averages that away.",
            "the_only_sanctioned_probe": "rotation applied to DELTANET ALONE, which is a "
                                         "different condition from the whole-model test "
                                         "that already failed. mlp and attention_gqa are "
                                         "near-Gaussian and stay closed.",
            "not_a_prediction": "mild excess kurtosis means there is something to "
                                "redistribute, not that redistributing it will pay.",
        },
        "stages": stages,
        "survivors_to_expand": survivors,
        "closed_cheaply": closed,
        "techniques_closed_by_probe": n_closed_tech,
        "remaining_frontier": frontier,
        "techniques_on_the_frontier": n_frontier_tech,
        "frontier_note": "these stages have techniques in the library and NO cheap "
                         "weights-only precondition, so they are where search is still "
                         "worth spending. They are untested, not endorsed.",
        "missing_probe_tensors": missing,
        "no_blind_sweep": "no stage is ranked by bit width; every verdict cites either a "
                          "measured precondition or a campaign receipt",
    }
    out["pass"] = bool(routed == len(lib) and stages and
                       (survivors or closed) and
                       sum(1 for v in stages.values()
                           if v["verdict"] == "ADJUDICATED_BY_CAMPAIGN_MEASUREMENT") >= 4)
    p = RH / "DOCTOR_TOURNAMENT.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"library {len(lib)} techniques, {routed} routed, "
          f"{len(out['unrouted'])} unrouted\n")
    print("diagnosis:")
    for o, d in diagnosis.items():
        print(f"  {o:14s} kurt={d['mean_excess_kurtosis']:>9.2f}  "
              f"outliers={d['mean_outlier_frac']:.5f}  "
              f"r90/full={d['mean_r90_over_full_rank']:.3f}  "
              f"deadrows={d['mean_near_zero_row_frac']:.5f}  "
              f"xlayer_cos={d['max_cross_layer_cosine']}")
    print()
    for st in CANONICAL_ORDER:
        v = stages[st]
        print(f"  {st:13s} n={v['n_techniques']:2d}  {v['verdict']}")
    print()
    for o, v in per_organ_coordinates.items():
        print(f"  COORDINATES {o:14s} kurt={v['excess_kurtosis']:>6.2f} "
              f"holds={v['holds']}")
    print(f"\nprobe-supported expansion: {survivors or 'none'}")
    print(f"closed cheaply by probe: {closed} ({n_closed_tech} techniques)")
    print(f"remaining frontier: {frontier} ({n_frontier_tech} techniques)")
    if missing:
        print(f"missing probe tensors: {missing}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
