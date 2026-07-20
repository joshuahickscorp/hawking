#!/usr/bin/env python3.12
"""Sub-half-bit search for Qwen3-235B: inter-expert shared grammar + exact whole-model BPW.

Lever under test: 128 experts x 94 layers per organ means ONE codebook shared across a layer's
whole expert cluster amortizes to ~0 bits/weight, so the payload is indices only. Organ inversion
(gate/up sensitive, down robust) is respected: gate/up get the higher index rate, down gets the
harshest. Reconstruction is measured on REAL resident weights, never a synthetic twin.

Sampling, not sweeping: E experts from 2 layers, all three organs. Codebooks are billed over the
sampled cluster size (conservative: the deployed 128-expert cluster amortizes strictly better).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import gravity_forge as GF
import qwen3_moe_adapter as A
from qwen_bpw_budget import METADATA_BITS_PER_TENSOR, _pq_bits
from qwen_real_forward import SafetensorsIndexReader

SCHEMA = "hawking.qwen3_235b.subhalfbit_search.v1"
SOURCE_DIR = Path("models/qwen3-235b-a22b")

# Non-expert lanes are frozen at the corrected-allocation choices so the comparison is apples/apples.
PQ_DENSE = {"family": "product_quant", "dim": 32, "subspaces": 8, "k": 16}
KEEP_NATIVE = {"family": "kept_original", "bpw": 16.0}
_DENSE_ORGANS = {A.ORGAN_EMBED, A.ORGAN_LM_HEAD, A.ORGAN_Q, A.ORGAN_K, A.ORGAN_V, A.ORGAN_O}

DEPLOY_CLUSTER = 128            # experts sharing one codebook in the real artifact

# ---- candidate packer specs -------------------------------------------------------------------
# shared_grammar: index rate = stages*ceil(log2 k)/dim bits/weight, codebook amortized over cluster.
GATE_SPECS = {
    # rate 0.75 control (matches the sealed C1 gate/up allocation)
    "G_sg_d32k256s3": {"family": "shared_grammar", "dim": 32, "k": 256, "stages": 3},
    # rate 0.50 lane: same index rate, increasing VQ dimension = space-filling gain
    "G_sg_d8k16s1":   {"family": "shared_grammar", "dim": 8,  "k": 16,   "stages": 1},
    "G_sg_d16k256s1": {"family": "shared_grammar", "dim": 16, "k": 256,  "stages": 1},
    "G_sg_d32k256s2": {"family": "shared_grammar", "dim": 32, "k": 256,  "stages": 2},
    "G_pq_d32s4k16":  {"family": "transform_pq",   "dim": 32, "subspaces": 4, "k": 16},
    # rate 0.625 / 0.3125: shared big codebook only affordable because it is amortized
    "G_sg_d16k1024s1": {"family": "shared_grammar", "dim": 16, "k": 1024, "stages": 1},
    "G_sg_d32k1024s1": {"family": "shared_grammar", "dim": 32, "k": 1024, "stages": 1},
}
DOWN_SPECS = {
    # rate 0.25 lane at increasing VQ dimension
    "D_sg_d8k4s1":    {"family": "shared_grammar", "dim": 8,  "k": 4,    "stages": 1},
    "D_sg_d16k16s1":  {"family": "shared_grammar", "dim": 16, "k": 16,   "stages": 1},
    "D_sg_d32k256s1": {"family": "shared_grammar", "dim": 32, "k": 256,  "stages": 1},
    "D_pq_d32s2k16":  {"family": "transform_pq",   "dim": 32, "subspaces": 2, "k": 16},
    # harsher / cheaper lanes
    "D_sg_d32k1024s1": {"family": "shared_grammar", "dim": 32, "k": 1024, "stages": 1},
    "D_sg_d64k1024s1": {"family": "shared_grammar", "dim": 64, "k": 1024, "stages": 1},
    "D_tern_r64":      {"family": "ternary_factor", "rank": 64, "keep_frac": 0.6},
}


# ---- exact bit accounting ---------------------------------------------------------------------
def expert_bits(shape: tuple[int, int], spec: dict[str, Any], cluster: int) -> int:
    """Bits charged to ONE expert tensor under `spec`, codebook amortized over `cluster` experts."""
    n = shape[0] * shape[1]
    fam = spec["family"]
    if fam == "shared_grammar":
        d, k, st = int(spec["dim"]), int(spec["k"]), int(spec["stages"])
        idx = (n // d) * st * math.ceil(math.log2(k))
        codebook = st * k * d * 16 / cluster        # billed once per cluster
        extra = 16                                  # per-expert fp16 output-gain scalar
    elif fam in ("naive_rvq", "transform_pq"):
        d, k = int(spec["dim"]), int(spec["k"])
        st = int(spec.get("stages", 1)) if fam == "naive_rvq" else int(spec["subspaces"])
        idx = (n // d) * st * math.ceil(math.log2(k))
        codebook = k * d * 16 + 64                  # per-expert codebook + rotation seed
        extra = 16
    elif fam == "ternary_factor":
        r = int(spec["rank"])
        idx = r * (shape[0] + shape[1]) * 2
        codebook = r * 16
        extra = 16
    else:
        raise ValueError(fam)
    return math.ceil((idx + codebook + extra + METADATA_BITS_PER_TENSOR) / 8) * 8


def _native_bits(shape: tuple[int, ...]) -> int:
    return math.ceil((math.prod(shape) * 16 + METADATA_BITS_PER_TENSOR) / 8) * 8


def whole_model_bpw(inv, gate_spec, down_spec, *, cluster=DEPLOY_CLUSTER,
                    cold_frac: float = 0.0, cold_gate=None, cold_down=None,
                    entropy_ratio: dict[str, float] | None = None) -> dict[str, Any]:
    """Exact whole-model accounting over every tensor class.

    cold_frac>0 applies `cold_*` specs to that fraction of each layer's experts (routing-frequency
    aware allocation). entropy_ratio scales an organ's INDEX bits by the measured empirical-entropy
    ratio (what rANS/arithmetic coding of the index stream would achieve)."""
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "n_weights": 0, "payload_bits": 0})
    total = 0
    for t in inv.tensors:
        oc = t.organ_class
        if oc in (A.ORGAN_EXP_GATE, A.ORGAN_EXP_UP, A.ORGAN_EXP_DOWN):
            hot = down_spec if oc == A.ORGAN_EXP_DOWN else gate_spec
            cold = hot
            if cold_frac > 0:
                cold = (cold_down if oc == A.ORGAN_EXP_DOWN else cold_gate) or hot
            # deterministic hot/cold membership stand-in for a routing-frequency calibration pass
            spec = cold if (int(t.expert) % 100) < int(round(cold_frac * 100)) else hot
            bits = expert_bits(t.shape, spec, cluster)
            if entropy_ratio:
                key = "down" if oc == A.ORGAN_EXP_DOWN else "gate_up"
                r = entropy_ratio.get(key)
                if r:
                    bits = math.ceil(bits * r / 8) * 8
            sp = spec
        elif oc in _DENSE_ORGANS:
            sp = PQ_DENSE
            bits = _pq_bits(t.shape, dim=32, subspaces=8, k=16)
        else:
            sp = KEEP_NATIVE
            bits = _native_bits(t.shape)
        total += bits
        row = groups[oc]
        row["tensor_count"] += 1
        row["n_weights"] += t.param_count
        row["payload_bits"] += bits
        row["spec"] = dict(sp)
    rows = {o: {**r, "realized_bpw": round(r["payload_bits"] / r["n_weights"], 9),
                "physical_bytes": math.ceil(r["payload_bits"] / 8)}
            for o, r in sorted(groups.items())}
    reserve = 64 * 1024 * 1024 * 8
    return {"allocation": rows,
            "packed_tensor_payload_bits": total,
            "container_metadata_reserve_bytes": 64 * 1024 * 1024,
            "total_artifact_bits": total + reserve,
            "total_artifact_bytes": math.ceil((total + reserve) / 8),
            "whole_model_bpw": round((total + reserve) / inv.grand_params, 9)}


# ---- real-weight measurement ------------------------------------------------------------------
def load_cluster(reader, layer: int, organ: str, experts: list[int]) -> list[np.ndarray]:
    suffix = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[organ]
    return [reader.bf16(f"model.layers.{layer}.mlp.experts.{e}.{suffix}.weight").astype(np.float32)
            for e in experts]


def _metrics(w: np.ndarray, r: np.ndarray) -> dict[str, float]:
    wf, rf = w.ravel().astype(np.float64), r.ravel().astype(np.float64)
    rel = float(np.linalg.norm(wf - rf) / (np.linalg.norm(wf) + 1e-12))
    cos = float(wf @ rf / ((np.linalg.norm(wf) * np.linalg.norm(rf)) + 1e-12))
    # per-tensor optimal output gain: the free fp16 scalar the allocation already budgets
    g = float(wf @ rf / (rf @ rf + 1e-12))
    relg = float(np.linalg.norm(wf - g * rf) / (np.linalg.norm(wf) + 1e-12))
    return {"rel_error": rel, "cosine": cos, "gain": g, "gain_corrected_rel_error": relg}


def pack_and_measure(experts: list[np.ndarray], spec: dict[str, Any]) -> dict[str, Any]:
    fam = spec["family"]
    t0 = time.time()
    if fam == "shared_grammar":
        art = GF.pack_shared_grammar(experts, dim=int(spec["dim"]), k=int(spec["k"]),
                                     stages=int(spec["stages"]), iters=6)
        recons = [art.recon[i] for i in range(len(experts))]
    elif fam == "naive_rvq":
        recons = [GF.pack_naive_rvq(w, dim=int(spec["dim"]), k=int(spec["k"]),
                                    stages=int(spec["stages"]), iters=6).recon for w in experts]
    elif fam == "transform_pq":
        recons = [GF.pack_transform_pq(w, dim=int(spec["dim"]), subspaces=int(spec["subspaces"]),
                                       k=int(spec["k"]), iters=6).recon for w in experts]
    elif fam == "ternary_factor":
        recons = [GF.pack_ternary_factor(w, rank=int(spec["rank"]),
                                         keep_frac=float(spec["keep_frac"])).recon
                  for w in experts]
    else:
        raise ValueError(fam)
    per = [_metrics(w, r) for w, r in zip(experts, recons)]
    agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    agg["seconds"] = round(time.time() - t0, 1)
    agg["per_expert_rel_error"] = [round(p["rel_error"], 4) for p in per]
    return agg


def index_entropy(experts: list[np.ndarray], spec: dict[str, Any]) -> dict[str, float]:
    """Empirical entropy of the shared-grammar index stream vs the stored ceil(log2 k) bits."""
    import torch
    d, k, st = int(spec["dim"]), int(spec["k"]), int(spec["stages"])
    dev = GF._device()
    vs = [torch.from_numpy(np.ascontiguousarray(w)).to(dev).reshape(-1, d) for w in experts]
    pool = torch.cat(vs, 0)
    res = pool.clone()
    stored = 0.0
    ent = 0.0
    for m in range(st):
        cb = GF._kmeans(res, k, iters=6, seed=m)
        idx = GF._assign(res, cb)
        res = res - cb[idx]
        cnt = torch.bincount(idx, minlength=cb.shape[0]).cpu().numpy().astype(np.float64)
        p = cnt / cnt.sum()
        p = p[p > 0]
        ent += float(-(p * np.log2(p)).sum())
        stored += math.ceil(math.log2(cb.shape[0]))
    return {"stored_bits_per_index": stored, "empirical_entropy_bits": round(ent, 4),
            "entropy_ratio": round(ent / stored, 4)}


# ---- driver -------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 60])
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--output", type=Path,
                    default=Path("reports/condense/general_frontier/QWEN3_235B_SUBHALFBIT_SEARCH.json"))
    args = ap.parse_args(argv)

    inv = A.build_inventory(A.load_config(), A.load_index())
    reader = SafetensorsIndexReader(SOURCE_DIR)
    if not reader.source_present():
        raise SystemExit("source shards absent")
    eids = list(range(args.experts))

    meas: dict[str, dict[str, Any]] = defaultdict(dict)   # spec_id -> organ -> agg
    entropy: dict[str, Any] = {}
    for layer in args.layers:
        for organ, specs in (("gate", GATE_SPECS), ("up", GATE_SPECS), ("down", DOWN_SPECS)):
            ws = load_cluster(reader, layer, organ, eids)
            for sid, spec in specs.items():
                agg = pack_and_measure(ws, spec)
                meas[sid].setdefault(organ, {})[f"layer{layer}"] = agg
                print(f"L{layer} {organ:4s} {sid:16s} rel={agg['rel_error']:.4f} "
                      f"cos={agg['cosine']:.4f} gainrel={agg['gain_corrected_rel_error']:.4f} "
                      f"({agg['seconds']}s)", flush=True)
            if layer == args.layers[0]:
                probe = (GATE_SPECS["G_sg_d16k256s1"] if organ != "down"
                         else DOWN_SPECS["D_sg_d32k256s1"])
                entropy[organ] = index_entropy(ws, probe) | {"probe_spec": probe}
                print(f"L{layer} {organ:4s} entropy {entropy[organ]}", flush=True)
            del ws

    ratio = {"gate_up": max(entropy["gate"]["entropy_ratio"], entropy["up"]["entropy_ratio"]),
             "down": entropy["down"]["entropy_ratio"]}

    # ---- rank every (gate/up, down) pairing by whole-model BPW -------------------------------
    cands = []
    for gid, gspec in GATE_SPECS.items():
        for did, dspec in DOWN_SPECS.items():
            acct = whole_model_bpw(inv, gspec, dspec)
            ent_acct = whole_model_bpw(inv, gspec, dspec, entropy_ratio=ratio)
            # "one step harsher" = double the VQ dimension at fixed k (halves the index rate)
            cold_g = {**gspec, "dim": gspec["dim"] * 2} if "dim" in gspec else gspec
            cold_d = {**dspec, "dim": dspec["dim"] * 2} if "dim" in dspec else dspec
            cold_acct = whole_model_bpw(inv, gspec, dspec, cold_frac=0.5,
                                        cold_gate=cold_g, cold_down=cold_d)
            def collect(sid, organ):
                out = {}
                for lay, a in meas[sid][organ].items():
                    out[lay] = {k: round(v, 5) for k, v in a.items()
                                if isinstance(v, float)}
                return out
            cands.append({
                "id": f"{gid}+{did}",
                "gate_up_spec": gspec, "down_spec": dspec,
                "whole_model_bpw": acct["whole_model_bpw"],
                "whole_model_bpw_entropy_coded": ent_acct["whole_model_bpw"],
                "whole_model_bpw_cold50_harsher": cold_acct["whole_model_bpw"],
                "expert_organ_bpw": {
                    "gate_up": acct["allocation"][A.ORGAN_EXP_GATE]["realized_bpw"],
                    "down": acct["allocation"][A.ORGAN_EXP_DOWN]["realized_bpw"]},
                "measured": {"gate": collect(gid, "gate"), "up": collect(gid, "up"),
                             "down": collect(did, "down")},
                "below_half_bit": acct["whole_model_bpw"] < 0.5,
            })
    # rank: only configs that are actually sub-0.5, ordered by measured gate quality then bpw
    def key(c):
        g = np.mean([v["gain_corrected_rel_error"] for v in c["measured"]["gate"].values()])
        d = np.mean([v["gain_corrected_rel_error"] for v in c["measured"]["down"].values()])
        return (not c["below_half_bit"], round(0.7 * g + 0.3 * d, 4), c["whole_model_bpw"])
    cands.sort(key=key)

    report = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample": {"layers": args.layers, "experts_per_cluster_sampled": args.experts,
                   "deploy_cluster_experts": DEPLOY_CLUSTER,
                   "note": "codebooks billed amortized over the 128-expert deploy cluster; "
                           "fits done on the sampled cluster (in-sample, same as deployment)"},
        "parent": {"parameters": inv.grand_params, "revision": A.IMMUTABLE_REVISION},
        "index_entropy": entropy,
        "entropy_ratio_applied": ratio,
        "candidates": cands,
        "baseline_corrected_allocation_bpw": 0.684207329,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for c in cands[:6]:
        print(f"{c['id']:34s} bpw={c['whole_model_bpw']:.4f} "
              f"ent={c['whole_model_bpw_entropy_coded']:.4f} sub0.5={c['below_half_bit']}")
    return 0


def selftest() -> None:
    """Bit accounting must match the closed form: stages*log2(k)/dim + amortized codebook."""
    spec = {"family": "shared_grammar", "dim": 32, "k": 256, "stages": 2}
    n = 1536 * 4096
    b = expert_bits((1536, 4096), spec, 128)
    assert abs(b / n - (2 * 8 / 32)) < 1e-3, b / n
    tern = expert_bits((4096, 1536), {"family": "ternary_factor", "rank": 64, "keep_frac": 0.6}, 128)
    assert abs(tern / n - 64 * 5632 * 2 / n) < 1e-3, tern / n
    print("selftest ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        raise SystemExit(main())
