#!/usr/bin/env python3
"""ExpertFamilyGenome — what is DIFFERENT about a routed model.

Directive §36-§38. Measured on real weights and real routed activations:
expert similarity, shared subspace, expert deltas, routing entropy, active experts per
token, hot/cold experts, and COMPLETE versus ACTIVE EBPW and bytes per token.

It also tests the MoE noetic hypothesis (§37) directly: is Expert_i approximately a
shared substrate plus a small expert-specific delta? Prior work on a different MoE found
experts mutually orthogonal, which would refute it. That was a different model; this
measures THIS one rather than inheriting the answer.
"""
import argparse, json, subprocess, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load(md, idx, name):
    from safetensors import safe_open
    with safe_open(md / idx[name], framework="pt") as f:
        return f.get_tensor(name).float().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    md = Path(a.model_dir)
    cfg = json.load(open(md / "config.json"))
    idx = json.load(open(md / "model.safetensors.index.json"))["weight_map"]
    cap = json.load(open(Path(a.capture) / "CAPTURE.json"))
    X = np.load(Path(a.capture) / f"X_layer{a.layer}.npy").astype(np.float32)

    t0 = time.time()
    E, K = a.experts, cfg["num_experts_per_tok"]
    pre = f"model.layers.{a.layer}.mlp.experts."
    gate = np.stack([load(md, idx, f"{pre}{e}.gate_proj.weight") for e in range(E)])
    up = np.stack([load(md, idx, f"{pre}{e}.up_proj.weight") for e in range(E)])
    down = np.stack([load(md, idx, f"{pre}{e}.down_proj.weight") for e in range(E)])
    router = load(md, idx, f"model.layers.{a.layer}.mlp.gate.weight")

    def flat(W):
        return W.reshape(W.shape[0], -1)

    def pairwise_cos(W):
        F = flat(W)
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        C = F @ F.T
        iu = np.triu_indices(len(F), 1)
        return C[iu]

    sim = {}
    for nm, W in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        c = pairwise_cos(W)
        sim[nm] = {"mean_cosine": float(c.mean()), "max_cosine": float(c.max()),
                   "p95_cosine": float(np.percentile(c, 95)),
                   "n_pairs": int(c.size)}

    # Shared subspace: how much of the stacked experts' energy lives in the top few
    # components shared across all of them.
    shared = {}
    for nm, W in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        F = flat(W).astype(np.float64)
        F = F - F.mean(0, keepdims=True)
        # economical: singular values of the E x P matrix of flattened experts
        s = np.linalg.svd(F, compute_uv=False)
        e2 = s ** 2
        tot = e2.sum()
        shared[nm] = {"energy_in_top1": float(e2[0] / tot),
                      "energy_in_top4": float(e2[:4].sum() / tot),
                      "energy_in_top8": float(e2[:8].sum() / tot),
                      "n_components": int(len(s)),
                      "uniform_reference_top8": round(8.0 / len(s), 6)}

    # The noetic hypothesis, stated as a measurement: subtract the mean expert and see how
    # much of each expert's norm is left. A small residual would mean shared substrate plus
    # tiny delta; a residual near 1.0 means the experts are essentially independent.
    noetic = {}
    for nm, W in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        F = flat(W).astype(np.float64)
        mu = F.mean(0, keepdims=True)
        resid = np.linalg.norm(F - mu, axis=1) / (np.linalg.norm(F, axis=1) + 1e-12)
        noetic[nm] = {"mean_residual_norm_after_shared_mean": float(resid.mean()),
                      "min": float(resid.min()), "max": float(resid.max())}

    # Routing on the real captured activations.
    logits = X @ router.T
    order = np.argsort(-logits, axis=1)[:, :K]
    counts = np.bincount(order.reshape(-1), minlength=cfg["num_experts"]).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    hot = np.argsort(-counts)[:8].tolist()
    cold = int((counts == 0).sum())

    params_per_expert = gate[0].size + up[0].size + down[0].size
    total_expert_params = params_per_expert * cfg["num_experts"]
    active_expert_params = params_per_expert * K
    genome = {
        "schema": "hawking.headless.expert_family_genome.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/expert_family_genome.py",
        "obligation": "G027 — MODEL_2_NEW_SCIENCE / ExpertFamilyGenome (directive §36-§38)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "specimen": str(md), "layer": a.layer,
        "n_experts_in_model": cfg["num_experts"], "n_experts_sampled": E,
        "num_experts_per_tok": K,
        "activations": {"real_not_synthetic": True, "n_tokens": int(X.shape[0]),
                        "exact_because": cap["exact_because"]},
        "expert_similarity": sim,
        "shared_subspace": shared,
        "noetic_hypothesis_test": {
            "hypothesis": "Expert_i ~= shared substrate + tiny expert-specific delta",
            "measurement": "fraction of each expert's norm remaining after subtracting the "
                           "mean expert across the sampled family",
            "per_tensor": noetic,
            "verdict": None,
        },
        "routing": {"entropy_bits": round(ent, 4),
                    "max_entropy_bits": round(float(np.log2(cfg["num_experts"])), 4),
                    "normalized_entropy": round(ent / float(np.log2(cfg["num_experts"])), 4),
                    "hot_experts": hot, "n_cold_experts_unrouted": cold,
                    "n_tokens": int(X.shape[0]),
                    "caveat": "routing measured on 112 captured tokens; a cold expert here "
                              "means unrouted on this sample, not unused by the model"},
        "complete_vs_active": {
            "params_per_expert": int(params_per_expert),
            "total_expert_params": int(total_expert_params),
            "active_expert_params_per_token": int(active_expert_params),
            "active_fraction": round(active_expert_params / total_expert_params, 6),
            "law": "an MoE can be physically excellent without a tiny COMPLETE EBPW if "
                   "little state is touched per token (directive §38)"},
        "wall_s": round(time.time() - t0, 1),
    }
    resid = float(np.mean([noetic[k]["mean_residual_norm_after_shared_mean"] for k in noetic]))
    genome["noetic_hypothesis_test"]["mean_residual_across_tensors"] = round(resid, 5)
    genome["noetic_hypothesis_test"]["verdict"] = (
        "REFUTED on this specimen: subtracting the shared mean expert removes almost none of "
        "each expert's norm, so there is no cheap shared substrate to factor out"
        if resid > 0.95 else
        "SUPPORTED: a shared substrate accounts for a material share of each expert"
        if resid < 0.7 else
        "PARTIAL: the shared mean removes a measurable but minority share")
    Path(a.emit).write_text(json.dumps(genome, indent=1))
    print(json.dumps({"similarity": {k: round(v["mean_cosine"], 5) for k, v in sim.items()},
                      "shared_top8_energy": {k: round(v["energy_in_top8"], 4)
                                             for k, v in shared.items()},
                      "uniform_reference_top8": shared["gate_proj"]["uniform_reference_top8"],
                      "noetic_residual": resid,
                      "verdict": genome["noetic_hypothesis_test"]["verdict"][:90],
                      "routing": genome["routing"]["normalized_entropy"],
                      "cold_experts": cold,
                      "active_fraction": genome["complete_vs_active"]["active_fraction"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
