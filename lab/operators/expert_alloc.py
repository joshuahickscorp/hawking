"""MoE per-expert bit allocation (absorbed from tools/condense/expert.py).

Kept: synthetic_experts, grid_quant_rel_l2, decide (verdict bands),
propose_allocation (hot 2-bit / cold floor / protected high-bit),
amortized_eff_bpw, measure_organs_from_weights (safetensors organs, no HF).

Dropped: HF AutoModel route-hook path (transformers), expert_cache_policy
simulation CLI, baker ladder import. Cache-policy is a pure projection for
OOC paging — not needed by current Q30 repack operators.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from typing import Any, Mapping, Sequence

from lab.operators.mixed_precision_alloc import BPW, grid_quant_rel_l2

UNIFORM_COV = 0.05
NONUNIFORM_RATIO = 1.5


def amortized_eff_bpw(alloc_bits: Mapping[str, int], elems: Mapping[str, int]) -> float:
    tot = sum(int(v) for v in elems.values())
    if tot == 0:
        return float("nan")
    return sum(BPW[int(alloc_bits[k])] * int(elems[k]) for k in alloc_bits) / tot


def synthetic_experts(
    n: int = 64,
    mode: str = "nonuniform",
    seed: int = 0,
    bits_set: Sequence[int] = (1, 2),
) -> list[dict[str, Any]]:
    """Fabricate a per-expert {route_freq, rel_l2@k} distribution for self-test."""
    rng = random.Random(seed)
    bits_set = sorted(int(b) for b in bits_set)
    experts: list[dict[str, Any]] = []
    if mode == "uniform":
        for e in range(n):
            base = 0.30 + rng.uniform(-0.006, 0.006)
            experts.append(
                {
                    "expert": e,
                    "route_freq": 1.0 / n,
                    "rel_l2": {b: base * (2.0 if b == min(bits_set) else 1.0) for b in bits_set},
                    "elems": 1_000_000,
                }
            )
        return experts
    raw = [1.0 / (1.0 + i) ** 1.1 for i in range(n)]
    rng.shuffle(raw)
    s = sum(raw)
    for e in range(n):
        rf = raw[e] / s
        hot = rf * n
        base2 = 0.08 + 0.55 * min(1.0, hot) + rng.uniform(0, 0.03)
        experts.append(
            {
                "expert": e,
                "route_freq": rf,
                "rel_l2": {
                    b: min(0.99, base2 * (2.1 if b == min(bits_set) else 1.0))
                    for b in bits_set
                },
                "elems": 1_000_000,
            }
        )
    return experts


def _stats(vals: Sequence[float]) -> dict[str, float]:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    if not clean:
        return dict(
            n=0,
            mean=float("nan"),
            stdev=float("nan"),
            cov=float("nan"),
            min=float("nan"),
            max=float("nan"),
            ratio=float("nan"),
        )
    n = len(clean)
    mean = sum(clean) / n
    var = sum((v - mean) ** 2 for v in clean) / n
    sd = math.sqrt(var)
    lo, hi = min(clean), max(clean)
    return dict(
        n=n,
        mean=mean,
        stdev=sd,
        cov=(sd / mean if mean else float("nan")),
        min=lo,
        max=hi,
        ratio=(hi / lo if lo > 0 else float("inf")),
    )


def decide(experts: Sequence[Mapping[str, Any]], bits_set: Sequence[int]) -> dict[str, Any]:
    """Verdict on whether per-expert allocation can beat uniform."""
    bits_set = sorted(int(b) for b in bits_set)
    floor = min(bits_set)
    rel_floor = [float(e["rel_l2"].get(floor, float("nan"))) for e in experts]
    st = _stats(rel_floor)
    imp = [
        float(e["rel_l2"].get(floor) or 0.0) * float(e.get("route_freq") or 0.0)
        for e in experts
    ]
    st_imp = _stats([v for v in imp if v > 0])

    nonuniform = (st["ratio"] >= NONUNIFORM_RATIO) and (st["cov"] >= UNIFORM_COV)
    uniform = st["cov"] < UNIFORM_COV
    if uniform:
        verdict = "UNIFORM"
        decision = (
            f"KILL SUBBIT-4: per-expert sensitivity is uniform (CoV {st['cov']:.3f} < "
            f"{UNIFORM_COV:.2f}) — dies the dense mixed-precision death. Do NOT build "
            f"the per-expert writer."
        )
        alive: bool | None = False
    elif nonuniform:
        verdict = "NON-UNIFORM"
        decision = (
            f"MoE sub-bit ALIVE: spread max/min = {st['ratio']:.2f}x "
            f"(>= {NONUNIFORM_RATIO:.1f}x), CoV {st['cov']:.3f} — cold experts tolerate "
            f"sub-bit while hot resist. Per-expert allocation can beat uniform."
        )
        alive = True
    else:
        verdict = "INCONCLUSIVE"
        decision = (
            f"INCONCLUSIVE: spread {st['ratio']:.2f}x, CoV {st['cov']:.3f} sits between "
            f"kill (<{UNIFORM_COV:.2f} CoV) and alive (>={NONUNIFORM_RATIO:.1f}x ratio) gates."
        )
        alive = None
    return dict(
        floor_bit=floor,
        spread_raw=st,
        spread_importance=st_imp,
        verdict=verdict,
        alive=alive,
        decision=decision,
    )


def propose_allocation(
    experts: Sequence[Mapping[str, Any]],
    bits_set: Sequence[int],
) -> tuple[dict[str, int], dict[str, int], float, dict[str, Any]]:
    """Hot experts → 2-bit (if available), cold → floor; router/shared protected high-bit."""
    bits_set = sorted(int(b) for b in bits_set)
    floor = min(bits_set)
    hot_bit = 2 if 2 in bits_set else max(bits_set)
    protect_bit = max(list(bits_set) + [4])
    rfs = sorted(float(e.get("route_freq") or 0.0) for e in experts)
    med = rfs[len(rfs) // 2] if rfs else 0.0
    alloc: dict[str, int] = {}
    elems: dict[str, int] = {}
    n_hot = n_cold = 0
    for e in experts:
        key = f"{e.get('layer', 'L')}.expert{e['expert']}"
        rf = float(e.get("route_freq") or 0.0)
        if rf > med:
            alloc[key] = hot_bit
            n_hot += 1
        else:
            alloc[key] = floor
            n_cold += 1
        elems[key] = int(e.get("elems") or 1)
    exp_total = sum(elems.values())
    prot_elems = int(0.03 * exp_total) or 1
    alloc["__protected_router_shared_attn"] = protect_bit
    elems["__protected_router_shared_attn"] = prot_elems
    amo = amortized_eff_bpw(alloc, elems)
    summary = dict(
        hot_experts=n_hot,
        hot_bit=hot_bit,
        cold_experts=n_cold,
        cold_bit=floor,
        protected_bit=protect_bit,
        protected_frac=round(prot_elems / (exp_total + prot_elems), 4),
        amortized_eff_bpw=round(amo, 3),
        note=(
            "amortized_eff_bpw uses BPW table (RHT+outlier folded); "
            "this is a PLAN, not a measured artifact."
        ),
    )
    return alloc, elems, amo, summary


def measure_organs_from_weights(
    organ_weights: Mapping[str, Any],
    *,
    bits_set: Sequence[int] = (1, 2),
    route_freq: Mapping[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Build expert rows from real organ weights.

    organ_weights keys: tensor names containing `.experts.<id>.` and ending in
    `_proj.weight` (Qwen3-MoE layout). route_freq optional {expert_id: freq};
    when absent, equal frequency is assumed (rel_L2-only spread still decides).
    """
    import torch

    bits_set = sorted(int(b) for b in bits_set)
    by_le: dict[tuple[str, int], list[dict[int, float]]] = {}
    elems_le: dict[tuple[str, int], int] = {}
    for name, W in organ_weights.items():
        if ".experts." not in name or not name.endswith("_proj.weight"):
            continue
        pre, rest = name.split(".experts.", 1)
        try:
            e = int(rest.split(".", 1)[0])
        except ValueError:
            continue
        t = W if isinstance(W, torch.Tensor) else torch.as_tensor(W)
        t = t.float()
        if t.ndim != 2 or min(t.shape) < 8:
            continue
        rel = {b: grid_quant_rel_l2(t, b) for b in bits_set}
        by_le.setdefault((pre, e), []).append(rel)
        elems_le[(pre, e)] = elems_le.get((pre, e), 0) + int(t.numel())

    experts: list[dict[str, Any]] = []
    for (pre, e), rels in sorted(by_le.items()):
        merged = {b: sum(r[b] for r in rels) / len(rels) for b in bits_set}
        if route_freq is not None and e in route_freq:
            rf = float(route_freq[e])
        else:
            rf = 1.0 / max(1, len(by_le))
        experts.append(
            {
                "layer": pre,
                "expert": e,
                "route_freq": rf,
                "rel_l2": merged,
                "elems": elems_le[(pre, e)],
            }
        )
    return experts


def run_rung(
    experts: Sequence[Mapping[str, Any]],
    *,
    bits_set: Sequence[int] = (1, 2),
) -> dict[str, Any]:
    bits_set = sorted(int(b) for b in bits_set)
    verdict = decide(experts, bits_set)
    alloc, elems, amo, summary = propose_allocation(experts, bits_set)
    return {
        "schema": "hawking.doctor.expert_alloc.v1",
        "rung": "expert_alloc",
        "layer": 2,
        "bits_set": list(bits_set),
        "n_experts": len(experts),
        "verdict": verdict,
        "allocation": alloc,
        "elems": elems,
        "amortized_eff_bpw": amo,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L2 expert allocation (live)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mode", default="nonuniform", choices=("nonuniform", "uniform"))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bits", default="1,2")
    args = ap.parse_args(argv)
    bits = [int(x) for x in args.bits.split(",") if x.strip()]
    if args.synthetic:
        experts = synthetic_experts(n=args.n, mode=args.mode, bits_set=bits)
        out = run_rung(experts, bits_set=bits)
        print(json.dumps(out, indent=2))
        return 0
    ap.error("pass --synthetic or use lab.operators.doctor_ladder")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
